// SPDX-License-Identifier: Apache-2.0

use crate::config::{ConnectionConfig, DeviceConfig, ModbusConfig, Parity};
use crate::models::InverterData;
use crate::mqtt::MqttMessage;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sunspec::client::{AsyncClient, AsyncDevice, Config as SunSpecConfig};
use sunspec::models::model1::Model1;
use sunspec::models::{
    model101::Model101, model102::Model102, model103::Model103, model111::Model111,
    model112::Model112, model113::Model113,
};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio_modbus::Slave;
use tokio_modbus::client::{Context as ModbusContext, Reader, rtu, tcp};
use tokio_modbus::slave::SlaveContext;
use tokio_rustls::TlsConnector;
use tokio_rustls::rustls::{ClientConfig, pki_types::ServerName};
use tokio_serial::SerialStream;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, error, info, info_span, trace, warn};

pub struct ConnectionTask {
    config: ConnectionConfig,
    mqtt_tx: mpsc::Sender<MqttMessage>,
    topic_prefix: String,
    ha_prefix: String,
    token: CancellationToken,
    root_cert_store: Arc<rustls::RootCertStore>,
}

struct DeviceState {
    config: DeviceConfig,
    last_poll: Option<Instant>,
    last_success_timestamp: Option<DateTime<Utc>>,
    serial: Option<String>,
    manufacturer: Option<String>,
    model: Option<String>,
    version: Option<String>,
    supported_model: Option<u16>,
    device: Option<AsyncDevice<Arc<Mutex<ModbusContext>>>>,
}

impl ConnectionTask {
    pub fn new(
        config: ConnectionConfig,
        mqtt_tx: mpsc::Sender<MqttMessage>,
        topic_prefix: String,
        ha_prefix: String,
        token: CancellationToken,
        root_cert_store: Arc<rustls::RootCertStore>,
    ) -> Self {
        Self {
            config,
            mqtt_tx,
            topic_prefix,
            ha_prefix,
            token,
            root_cert_store,
        }
    }

    pub async fn run(self) -> Result<()> {
        let span = info_span!("connection", name = %self.config.name);
        self.run_internal().instrument(span).await
    }

    async fn run_internal(self) -> Result<()> {
        info!("Starting connection task");

        let mut devices: Vec<DeviceState> = self
            .config
            .devices
            .iter()
            .map(|d| DeviceState {
                config: d.clone(),
                last_poll: None,
                last_success_timestamp: None,
                serial: None,
                manufacturer: None,
                model: None,
                version: None,
                supported_model: None,
                device: None,
            })
            .collect();

        let mut first_run = true;

        while !self.token.is_cancelled() {
            if !first_run {
                info!("Attempting to reconnect");
            }

            match self.establish_connection().await {
                Ok(ctx) => {
                    info!("Connected");
                    match self.initialize_devices(ctx, &mut devices).await {
                        Ok(_) => {
                            if self.token.is_cancelled() {
                                break;
                            }
                            if let Err(e) = self.run_polling_loop(&mut devices).await {
                                error!("Error in polling loop: {}. Reconnecting...", e);
                            }
                        }
                        Err(e) => {
                            error!(
                                "Failed to initialize devices: {}. Reconnecting in 10s...",
                                e
                            );
                            tokio::select! {
                                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
                                _ = self.token.cancelled() => break,
                            }
                        }
                    }
                }
                Err(e) => {
                    if self.token.is_cancelled() {
                        break;
                    }
                    error!(
                        "Failed to establish connection: {}. Reconnecting in 10s...",
                        e
                    );
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(10)) => {}
                        _ = self.token.cancelled() => break,
                    }
                }
            }
            first_run = false;
        }

        info!("Connection task finished cleanly");
        Ok(())
    }

    async fn initialize_devices(
        &self,
        ctx: ModbusContext,
        devices: &mut [DeviceState],
    ) -> Result<()> {
        let client = AsyncClient::new(ctx, SunSpecConfig::default());

        // Initial discovery and Model 1 poll
        for device_state in devices.iter_mut() {
            if self.token.is_cancelled() {
                return Ok(());
            }
            match client.device(device_state.config.unit_id).await {
                Ok(device) => {
                    // Find supported inverter model (do this first as it's needed for polling)
                    let supported = [101, 102, 103, 111, 112, 113];
                    let available_models = device.models.supported_model_ids();
                    for id in supported {
                        if available_models.contains(&id) {
                            device_state.supported_model = Some(id);
                            break;
                        }
                    }

                    // Try to read Model 1 for metadata/serial
                    match tokio::time::timeout(
                        Duration::from_secs(10),
                        device.read_model::<Model1>(),
                    )
                    .instrument(info_span!(
                        "model1_read",
                        unit_id = device_state.config.unit_id
                    ))
                    .await
                    {
                        Ok(Ok(m1)) => {
                            let serial = m1.sn.trim().to_string();
                            let manufacturer = m1.mn.trim().to_string();
                            let model = m1.md.trim().to_string();
                            let version_opt = m1
                                .vr
                                .as_ref()
                                .map(|v| v.trim().to_string())
                                .filter(|v| !v.is_empty());

                            if serial.is_empty() {
                                warn!(
                                    "Device {} returned an empty serial number, skipping discovery",
                                    device_state.config.unit_id
                                );
                                continue;
                            }

                            // Only log and publish discovery if it's new or changed
                            if device_state.serial.as_ref() != Some(&serial) {
                                info!(
                                    "Discovered device: {} {} (Serial: {})",
                                    manufacturer, model, serial
                                );
                                device_state.serial = Some(serial.clone());
                                device_state.manufacturer = Some(manufacturer.clone());
                                device_state.model = Some(model.clone());
                                device_state.version = version_opt.clone();
                                self.publish_discovery(
                                    &serial,
                                    &manufacturer,
                                    &model,
                                    version_opt.as_deref(),
                                    device_state.supported_model,
                                )
                                .await?;
                            } else {
                                info!("Refreshed connection to device (Serial: {})", serial);
                            }
                        }
                        Ok(Err(e)) => {
                            if device_state.serial.is_none() {
                                warn!(
                                    "Failed to read Model 1 from device {} and no previous serial known: {}",
                                    device_state.config.unit_id, e
                                );
                            } else {
                                info!(
                                    "Failed to refresh Model 1 for device, using cached info (Serial: {:?})",
                                    device_state.serial
                                );
                            }
                        }
                        Err(_) => {
                            warn!(
                                "Timeout reading Model 1 from device {}",
                                device_state.config.unit_id
                            );
                        }
                    }

                    // Store the device for later use
                    device_state.device = Some(device);
                }
                Err(e) => {
                    error!(
                        "Failed to discover device {}: {}",
                        device_state.config.unit_id, e
                    );
                }
            }
        }
        Ok(())
    }

    async fn establish_connection(&self) -> Result<ModbusContext> {
        match &self.config.modbus {
            ModbusConfig::Tcp { address, tls } => {
                info!("Connecting to Modbus TCP at {} (TLS: {})", address, tls);
                let stream = tokio::net::TcpStream::connect(address)
                    .await
                    .context(format!("Failed to connect to {}", address))?;

                if *tls {
                    let config = ClientConfig::builder()
                        .with_root_certificates(Arc::clone(&self.root_cert_store))
                        .with_no_client_auth();
                    let connector = TlsConnector::from(Arc::new(config));

                    let host = address
                        .split(':')
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("Invalid address"))?;
                    let server_name = ServerName::try_from(host)
                        .context("invalid server name")?
                        .to_owned();

                    let tls_stream = connector
                        .connect(server_name, stream)
                        .await
                        .context("TLS handshake failed")?;
                    Ok(tcp::attach(tls_stream))
                } else {
                    Ok(tcp::attach(stream))
                }
            }
            ModbusConfig::Rtu {
                device,
                baud_rate,
                parity,
            } => {
                info!(
                    "Connecting to Modbus RTU at {} ({} baud, parity: {:?})",
                    device, baud_rate, parity
                );
                let builder = tokio_serial::new(device, *baud_rate).parity(match parity {
                    Parity::None => tokio_serial::Parity::None,
                    Parity::Even => tokio_serial::Parity::Even,
                    Parity::Odd => tokio_serial::Parity::Odd,
                });

                let port = SerialStream::open(&builder)?;
                Ok(rtu::attach_slave(port, Slave(0)))
            }
        }
    }

    async fn run_polling_loop(&self, devices: &mut [DeviceState]) -> Result<()> {
        let mut last_activity = Instant::now();
        let keep_alive_interval = self.config.keep_alive_interval.unwrap_or(30);
        let keep_alive_duration = Duration::from_secs(keep_alive_interval);

        loop {
            if self.token.is_cancelled() {
                info!("Shutting down polling loop");
                return Ok(());
            }

            let now = Instant::now();
            // Start with next keep-alive as the upper bound if enabled
            let mut next_wakeup = if keep_alive_interval > 0 {
                last_activity + keep_alive_duration
            } else {
                now + Duration::from_secs(3600) // Default large sleep
            };

            for device_state in devices.iter() {
                let interval = Duration::from_secs(device_state.config.interval);

                let next_poll = device_state
                    .last_poll
                    .map(|last| last + interval)
                    .unwrap_or(now);

                next_wakeup = next_wakeup.min(next_poll);
            }

            let sleep_duration = next_wakeup
                .saturating_duration_since(Instant::now())
                .max(Duration::from_millis(10));

            trace!("Sleeping for {:?}", sleep_duration);

            tokio::select! {
                _ = tokio::time::sleep(sleep_duration) => {}
                _ = self.token.cancelled() => {
                    info!("Cancelled");
                    return Ok(());
                }
            }

            let now = Instant::now();

            // Check if we need to send a keep-alive "ping"
            if keep_alive_interval > 0 && now.duration_since(last_activity) >= keep_alive_duration {
                // Try to ping the first device that we've successfully discovered
                if let Some(state) = devices.iter().find(|d| d.device.is_some()) {
                    let ping_span = info_span!("keep_alive", unit_id = state.config.unit_id);
                    let device = state.device.as_ref().unwrap();
                    async {
                        info!("Sending keep-alive");

                        // Efficiently ping by reading only the first 2 registers of Model 1 (ID and Length)
                        // This is much more efficient than reading the whole Model 1.
                        let addr = device.models.m1.addr;
                        let mut ctx = device.client.lock().await;
                        ctx.set_slave(Slave(device.slave_id));
                        let _ = tokio::time::timeout(
                            Duration::from_secs(5),
                            ctx.read_holding_registers(addr, 2),
                        )
                        .await??;

                        Ok::<(), anyhow::Error>(())
                    }
                    .instrument(ping_span)
                    .await?;
                    last_activity = Instant::now();
                }
            }

            for device_state in devices.iter_mut() {
                if self.token.is_cancelled() {
                    break;
                }
                let interval = Duration::from_secs(device_state.config.interval);

                let next_poll = device_state
                    .last_poll
                    .map(|last| last + interval)
                    .unwrap_or(now);

                if next_poll <= now {
                    self.perform_device_poll(device_state, now).await?;
                    last_activity = Instant::now();
                }
            }
        }
    }

    async fn perform_device_poll(
        &self,
        device_state: &mut DeviceState,
        now: Instant,
    ) -> Result<()> {
        let span = info_span!("poll", unit_id = device_state.config.unit_id);
        self.perform_device_poll_internal(device_state, now)
            .instrument(span)
            .await
    }

    async fn perform_device_poll_internal(
        &self,
        device_state: &mut DeviceState,
        now: Instant,
    ) -> Result<()> {
        // ONLY poll if serial number and supported_model are known
        if let (Some(serial), Some(model_id), Some(device)) = (
            &device_state.serial,
            device_state.supported_model,
            &device_state.device,
        ) {
            let poll_result =
                tokio::time::timeout(Duration::from_secs(10), self.poll_device(device, model_id))
                    .await;

            match poll_result {
                Ok(Ok(data)) => {
                    info!("Successfully polled (Serial: {})", serial);
                    device_state.last_success_timestamp = Some(Utc::now());
                    let payload = serde_json::to_string(&data)?;
                    let topic = self.inverter_topic(serial);
                    self.mqtt_tx
                        .send(MqttMessage::Publish {
                            topic,
                            payload,
                            retain: false,
                        })
                        .await?;

                    let (status_topic, status_payload) = self.status_message(
                        serial,
                        "OK",
                        None,
                        device_state.last_success_timestamp.as_ref(),
                    );
                    self.mqtt_tx
                        .send(MqttMessage::Publish {
                            topic: status_topic,
                            payload: status_payload,
                            retain: false,
                        })
                        .await?;
                }
                Ok(Err(e)) => {
                    error!("Failed to poll: {}", e);
                    // Update status with error
                    let (status_topic, status_payload) = self.status_message(
                        serial,
                        "ERROR",
                        Some(&e.to_string()),
                        device_state.last_success_timestamp.as_ref(),
                    );
                    let _ = self
                        .mqtt_tx
                        .send(MqttMessage::Publish {
                            topic: status_topic,
                            payload: status_payload,
                            retain: false,
                        })
                        .await;
                }
                Err(_) => {
                    error!("Timeout polling");
                    let (status_topic, status_payload) = self.status_message(
                        serial,
                        "ERROR",
                        Some("Timeout"),
                        device_state.last_success_timestamp.as_ref(),
                    );
                    let _ = self
                        .mqtt_tx
                        .send(MqttMessage::Publish {
                            topic: status_topic,
                            payload: status_payload,
                            retain: false,
                        })
                        .await;
                }
            }
        }

        device_state.last_poll = Some(now);
        Ok(())
    }

    fn inverter_topic(&self, serial: &str) -> String {
        format!("{}/inverter/{}", self.topic_prefix, serial)
    }

    fn status_message(
        &self,
        serial: &str,
        status: &str,
        error: Option<&str>,
        last_success: Option<&DateTime<Utc>>,
    ) -> (String, String) {
        let topic = format!("{}/inverter/{}/status", self.topic_prefix, serial);
        let payload = serde_json::json!({
            "timestamp": last_success.map(|dt| dt.to_rfc3339()),
            "status": status,
            "error": error
        })
        .to_string();
        (topic, payload)
    }

    async fn poll_device(
        &self,
        device: &AsyncDevice<Arc<Mutex<ModbusContext>>>,
        model_id: u16,
    ) -> Result<InverterData> {
        match model_id {
            101 => Ok(device.read_model::<Model101>().await?.into()),
            102 => Ok(device.read_model::<Model102>().await?.into()),
            103 => Ok(device.read_model::<Model103>().await?.into()),
            111 => Ok(device.read_model::<Model111>().await?.into()),
            112 => Ok(device.read_model::<Model112>().await?.into()),
            113 => Ok(device.read_model::<Model113>().await?.into()),
            _ => anyhow::bail!("Unsupported model {}", model_id),
        }
    }

    async fn publish_discovery(
        &self,
        serial: &str,
        manufacturer: &str,
        model: &str,
        version: Option<&str>,
        model_id: Option<u16>,
    ) -> Result<()> {
        let mut sensors = vec![
            ("W", Some("W"), Some("power"), Some("measurement"), "Power"),
            (
                "WH",
                Some("Wh"),
                Some("energy"),
                Some("total_increasing"),
                "Energy",
            ),
            (
                "Hz",
                Some("Hz"),
                Some("frequency"),
                Some("measurement"),
                "Frequency",
            ),
            (
                "TmpCab",
                Some("°C"),
                Some("temperature"),
                Some("measurement"),
                "Cabinet Temperature",
            ),
            (
                "TmpSnk",
                Some("°C"),
                Some("temperature"),
                Some("measurement"),
                "Heat Sink Temperature",
            ),
            ("St", None, Some("enum"), None, "Status"),
        ];

        // Add phase-specific sensors based on model
        if let Some(id) = model_id {
            // All supported models have at least Phase A
            sensors.push((
                "PhVphA",
                Some("V"),
                Some("voltage"),
                Some("measurement"),
                "Voltage Phase A",
            ));
            sensors.push((
                "AphA",
                Some("A"),
                Some("current"),
                Some("measurement"),
                "Current Phase A",
            ));

            if matches!(id, 102 | 103 | 112 | 113) {
                sensors.push((
                    "PhVphB",
                    Some("V"),
                    Some("voltage"),
                    Some("measurement"),
                    "Voltage Phase B",
                ));
                sensors.push((
                    "AphB",
                    Some("A"),
                    Some("current"),
                    Some("measurement"),
                    "Current Phase B",
                ));
            }

            if matches!(id, 103 | 113) {
                sensors.push((
                    "PhVphC",
                    Some("V"),
                    Some("voltage"),
                    Some("measurement"),
                    "Voltage Phase C",
                ));
                sensors.push((
                    "AphC",
                    Some("A"),
                    Some("current"),
                    Some("measurement"),
                    "Current Phase C",
                ));
            }
        }

        for (name, unit, device_class, state_class, label) in sensors {
            let enabled_by_default = matches!(name, "W" | "WH" | "St");
            let options = if name == "St" {
                Some(vec![
                    "OFF",
                    "SLEEPING",
                    "STARTING",
                    "MPPT",
                    "THROTTLED",
                    "SHUTTING_DOWN",
                    "FAULT",
                    "STANDBY",
                    "UNKNOWN",
                ])
            } else {
                None
            };

            let ctx = DiscoveryContext {
                manufacturer,
                model,
                version,
                name,
                unit,
                device_class,
                state_class,
                label,
                enabled_by_default,
                options,
            };
            let (topic, payload) = self.discovery_message(serial, &ctx);
            self.mqtt_tx
                .send(MqttMessage::Publish {
                    topic,
                    payload,
                    retain: true,
                })
                .await?;
        }

        Ok(())
    }

    fn discovery_message(&self, serial: &str, ctx: &DiscoveryContext) -> (String, String) {
        let topic = format!("{}/sensor/{}/{}/config", self.ha_prefix, serial, ctx.name);
        let mut payload = serde_json::json!({
            "name": ctx.label,
            "state_topic": self.inverter_topic(serial),
            "value_template": format!("{{{{ value_json.{} }}}}", ctx.name),
            "unique_id": format!("{}_{}_{}", self.topic_prefix, serial, ctx.name),
            "force_update": true,
            "enabled_by_default": ctx.enabled_by_default,
            "device": {
                "identifiers": [serial],
                "name": format!("Inverter {}", serial),
                "manufacturer": ctx.manufacturer,
                "model": ctx.model,
            }
        });

        if let Some(v) = ctx.version {
            payload["device"]["sw_version"] = serde_json::json!(v);
        }

        if let Some(unit) = ctx.unit {
            payload["unit_of_measurement"] = serde_json::json!(unit);
        }
        if let Some(dc) = ctx.device_class {
            payload["device_class"] = serde_json::json!(dc);
        }
        if let Some(sc) = ctx.state_class {
            payload["state_class"] = serde_json::json!(sc);
        }
        if let Some(options) = &ctx.options {
            payload["options"] = serde_json::json!(options);
        }

        (topic, payload.to_string())
    }
}

struct DiscoveryContext<'a> {
    manufacturer: &'a str,
    model: &'a str,
    version: Option<&'a str>,
    name: &'a str,
    unit: Option<&'a str>,
    device_class: Option<&'a str>,
    state_class: Option<&'a str>,
    label: &'a str,
    enabled_by_default: bool,
    options: Option<Vec<&'static str>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn test_task() -> ConnectionTask {
        let (tx, _) = mpsc::channel(1);
        let root_cert_store = Arc::new(rustls::RootCertStore::empty());
        ConnectionTask {
            config: ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "127.0.0.1:502".to_string(),
                    tls: false,
                },
                devices: vec![],
                keep_alive_interval: None,
            },
            mqtt_tx: tx,
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
            token: CancellationToken::new(),
            root_cert_store,
        }
    }

    #[test]
    fn test_topics() {
        let task = test_task();
        assert_eq!(task.inverter_topic("SN123"), "solar/inverter/SN123");

        let (status_topic, payload) = task.status_message("SN123", "OK", None, None);
        assert_eq!(status_topic, "solar/inverter/SN123/status");
        assert!(payload.contains("\"timestamp\":null"));

        let now = Utc::now();
        let (_, payload_with_ts) = task.status_message("SN123", "OK", None, Some(&now));
        assert!(payload_with_ts.contains(&now.to_rfc3339()));
    }

    #[test]
    fn test_discovery_message() {
        let task = test_task();
        let ctx = DiscoveryContext {
            manufacturer: "Brand",
            model: "ModelX",
            version: Some("1.2.3"),
            name: "W",
            unit: Some("W"),
            device_class: Some("power"),
            state_class: Some("measurement"),
            label: "Power",
            enabled_by_default: true,
            options: None,
        };
        let (topic, payload) = task.discovery_message("SN123", &ctx);

        assert_eq!(topic, "homeassistant/sensor/SN123/W/config");
        assert!(payload.contains("\"state_topic\":\"solar/inverter/SN123\""));
        assert!(payload.contains("\"unique_id\":\"solar_SN123_W\""));
        assert!(payload.contains("\"manufacturer\":\"Brand\""));
        assert!(payload.contains("\"model\":\"ModelX\""));
        assert!(payload.contains("\"sw_version\":\"1.2.3\""));
        assert!(payload.contains("\"state_class\":\"measurement\""));
        assert!(payload.contains("\"force_update\":true"));
        assert!(payload.contains("\"enabled_by_default\":true"));
    }

    #[test]
    fn test_discovery_message_enum() {
        let task = test_task();
        let ctx = DiscoveryContext {
            manufacturer: "Brand",
            model: "ModelX",
            version: None,
            name: "St",
            unit: None,
            device_class: Some("enum"),
            state_class: None,
            label: "Status",
            enabled_by_default: true,
            options: Some(vec!["OFF", "ON"]),
        };
        let (topic, payload) = task.discovery_message("SN123", &ctx);

        assert_eq!(topic, "homeassistant/sensor/SN123/St/config");
        assert!(payload.contains("\"device_class\":\"enum\""));
        assert!(payload.contains("\"options\":[\"OFF\",\"ON\"]"));
        assert!(!payload.contains("\"sw_version\""));
        assert!(!payload.contains("\"unit_of_measurement\""));
        assert!(!payload.contains("\"state_class\""));
    }
}
