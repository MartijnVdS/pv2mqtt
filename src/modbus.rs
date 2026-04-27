// SPDX-License-Identifier: Apache-2.0

use crate::config::{ConnectionConfig, DeviceConfig, ModbusConfig, Parity};
use crate::error::{Pv2MqttError, Result};
use crate::models::{InverterData, SUPPORTED_MODELS};
use crate::mqtt::MqttMessage;
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
use tracing::{Instrument, debug, error, info, info_span, trace, warn};

const CONNECT_TIMEOUT_SECS: u64 = 10;
const RECONNECT_TIMEOUT_SECS: u64 = 10;

const READ_TIMEOUT_SECS: u64 = 10;
const POLL_TIMEOUT_SECS: u64 = 10;

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

impl DeviceState {
    fn clear_connection(&mut self) {
        self.device = None;
    }
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
            // Explicitly clear any old connection handles before attempting to reconnect.
            // This ensures that the previous socket/serial port is closed.
            for device_state in devices.iter_mut() {
                device_state.clear_connection();
            }

            if !first_run {
                info!("Attempting to reconnect");
            }

            let result: Result<()> = async {
                let ctx = self.establish_connection().await?;
                info!("Connected");
                self.run_polling_loop(ctx, &mut devices).await?;
                Ok(())
            }
            .await;

            if let Err(e) = result {
                error!(
                    "Error: {}. Reconnecting in {}s...",
                    e, RECONNECT_TIMEOUT_SECS
                );
                tokio::select! {
                    biased;
                    _ = self.token.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_secs(RECONNECT_TIMEOUT_SECS)) => {}
                }
            }
            first_run = false;
        }

        info!("Connection task finished cleanly");
        Ok(())
    }

    async fn run_polling_loop(
        &self,
        ctx: ModbusContext,
        devices: &mut [DeviceState],
    ) -> Result<()> {
        let client = AsyncClient::new(
            ctx,
            SunSpecConfig {
                // Disable read timeout in the SunSpec library
                // We have our own timeout for a poll cycle
                read_timeout: None,
                ..SunSpecConfig::default()
            },
        );

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
                    trace!("Cancelled");
                    return Ok(());
                }
            }

            let now = Instant::now();

            // Check if we need to send a keep-alive "ping"
            if keep_alive_interval > 0 && now.duration_since(last_activity) >= keep_alive_duration {
                // Try to ping the first device that we've successfully discovered
                if let Some((state, device)) = devices
                    .iter()
                    .find_map(|d| d.device.as_ref().map(|dev| (d, dev)))
                {
                    let ping_span = info_span!("keep_alive", unit_id = state.config.unit_id);
                    let ping_res: Result<()> = async {
                        debug!("Sending keep-alive");

                        // Efficiently ping by reading only the first 2 registers of Model 1 (ID and Length)
                        // This is much more efficient than reading the whole Model 1.
                        let addr = device.models.m1.addr;
                        let mut ctx = device.client.lock().await;
                        ctx.set_slave(Slave(device.slave_id));
                        let _ = tokio::time::timeout(
                            Duration::from_secs(READ_TIMEOUT_SECS),
                            ctx.read_holding_registers(addr, 2),
                        )
                        .await
                        .map_err(|_| {
                            Pv2MqttError::ModbusTimeout(format!(
                                "Keep-alive timeout for unit {}",
                                device.slave_id
                            ))
                        })??;

                        Ok(())
                    }
                    .instrument(ping_span)
                    .await;

                    if let Err(e) = ping_res {
                        error!("Keep-alive failed: {}", e);
                        return Err(e);
                    }
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
                    // Update last_poll before starting to ensure we don't immediately retry on timeout reconnect
                    device_state.last_poll = Some(now);

                    let res = if device_state.device.is_none() {
                        match self.discover_device(&client, device_state).await {
                            Ok(_) => {
                                // Successfully discovered, perform initial poll immediately
                                self.perform_device_poll(device_state, now).await
                            }
                            Err(e) => Err(e),
                        }
                    } else {
                        self.perform_device_poll(device_state, now).await
                    };

                    if let Err(e) = res {
                        if matches!(e, Pv2MqttError::ModbusTimeout(_)) {
                            return Err(e);
                        }
                        // Non-timeout errors are logged but don't break the connection
                        error!(
                            "Error processing device {}: {}",
                            device_state.config.unit_id, e
                        );
                    }

                    last_activity = Instant::now();
                }
            }
        }
    }

    async fn discover_device(
        &self,
        client: &AsyncClient<Arc<Mutex<ModbusContext>>>,
        device_state: &mut DeviceState,
    ) -> Result<()> {
        let unit_id = device_state.config.unit_id;
        let span = info_span!("discovery", unit_id);

        async {
            let device_res = tokio::time::timeout(
                Duration::from_secs(POLL_TIMEOUT_SECS),
                client.device(unit_id),
            )
            .await;

            let device = match device_res {
                Ok(Ok(d)) => d,
                Ok(Err(e)) => {
                    return Err(Pv2MqttError::DeviceDiscovery(unit_id, e.to_string()));
                }
                Err(_) => {
                    return Err(Pv2MqttError::ModbusTimeout(format!(
                        "Timeout discovering device {}",
                        unit_id
                    )));
                }
            };

            // Find supported inverter model
            let available_models = device.models.supported_model_ids();
            device_state.supported_model = SUPPORTED_MODELS
                .iter()
                .find(|&&id| available_models.contains(&id))
                .copied();

            if device_state.supported_model.is_none() {
                let available = available_models
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(Pv2MqttError::DeviceDiscovery(
                    unit_id,
                    format!(
                        "No supported inverter model found. Available: {}",
                        available
                    ),
                ));
            }

            // Try to read Model 1 for metadata/serial
            let m1_res = tokio::time::timeout(
                Duration::from_secs(POLL_TIMEOUT_SECS),
                device.read_model::<Model1>(),
            )
            .instrument(info_span!("model1_read", unit_id))
            .await;

            match m1_res {
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
                            unit_id
                        );
                        return Ok(());
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
                        return Err(Pv2MqttError::ModelRead(1, e.to_string()));
                    }
                    info!(
                        "Failed to refresh Model 1 for device, using cached info (Serial: {:?})",
                        device_state.serial
                    );
                }
                Err(_) => {
                    return Err(Pv2MqttError::ModbusTimeout(format!(
                        "Timeout reading Model 1 from device {}",
                        unit_id
                    )));
                }
            }

            // Store the device for later use
            device_state.device = Some(device);
            Ok(())
        }
        .instrument(span)
        .await
    }

    async fn establish_connection(&self) -> Result<ModbusContext> {
        match &self.config.modbus {
            ModbusConfig::Tcp { address, tls } => {
                info!("Connecting to Modbus TCP at {} (TLS: {})", address, tls);
                let stream = tokio::time::timeout(
                    Duration::from_secs(CONNECT_TIMEOUT_SECS),
                    tokio::net::TcpStream::connect(address),
                )
                .await
                .map_err(|_| {
                    Pv2MqttError::ModbusTcpConnection(format!("Timeout connecting to {}", address))
                })?
                .map_err(|e| {
                    Pv2MqttError::ModbusTcpConnection(format!(
                        "Failed to connect to {}: {}",
                        address, e
                    ))
                })?;

                if *tls {
                    let config = ClientConfig::builder()
                        .with_root_certificates(Arc::clone(&self.root_cert_store))
                        .with_no_client_auth();
                    let connector = TlsConnector::from(Arc::new(config));

                    let host = address.split(':').next().ok_or_else(|| {
                        Pv2MqttError::ModbusTcpConnection("Invalid address".to_string())
                    })?;
                    let server_name = ServerName::try_from(host)
                        .map_err(|e| {
                            Pv2MqttError::ModbusTcpConnection(format!("invalid server name: {}", e))
                        })?
                        .to_owned();

                    let tls_stream = tokio::time::timeout(
                        Duration::from_secs(CONNECT_TIMEOUT_SECS),
                        connector.connect(server_name, stream),
                    )
                    .await
                    .map_err(|_| {
                        Pv2MqttError::ModbusTcpConnection(format!(
                            "Timeout during TLS handshake with {}",
                            address
                        ))
                    })?
                    .map_err(|e| {
                        Pv2MqttError::ModbusTcpConnection(format!("TLS handshake failed: {}", e))
                    })?;
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
            let poll_result = tokio::time::timeout(
                Duration::from_secs(POLL_TIMEOUT_SECS),
                self.poll_device(device, model_id),
            )
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
                    let pv_err = Pv2MqttError::ModelRead(model_id, e.to_string());
                    error!("Failed to poll: {}", pv_err);
                    // Update status with error
                    let (status_topic, status_payload) = self.status_message(
                        serial,
                        "ERROR",
                        Some(pv_err.clone()),
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
                    return Err(pv_err);
                }
                Err(_) => {
                    let pv_err =
                        Pv2MqttError::ModbusTimeout(format!("Timeout polling device {}", serial));
                    error!("{}", pv_err);
                    let (status_topic, status_payload) = self.status_message(
                        serial,
                        "ERROR",
                        Some(pv_err.clone()),
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
                    return Err(pv_err);
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
        error: Option<Pv2MqttError>,
        last_success: Option<&DateTime<Utc>>,
    ) -> (String, String) {
        let topic = format!("{}/inverter/{}/status", self.topic_prefix, serial);
        let payload = serde_json::json!({
            "timestamp": last_success.map(|dt| dt.to_rfc3339()),
            "status": status,
            "error": error.as_ref().map(|e| e.to_string()),
            "error_category": error.as_ref().map(|e| e.category()),
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
            101 => Ok(device
                .read_model::<Model101>()
                .await
                .map_err(|e| Pv2MqttError::ModelRead(101, e.to_string()))?
                .into()),
            102 => Ok(device
                .read_model::<Model102>()
                .await
                .map_err(|e| Pv2MqttError::ModelRead(102, e.to_string()))?
                .into()),
            103 => Ok(device
                .read_model::<Model103>()
                .await
                .map_err(|e| Pv2MqttError::ModelRead(103, e.to_string()))?
                .into()),
            111 => Ok(device
                .read_model::<Model111>()
                .await
                .map_err(|e| Pv2MqttError::ModelRead(111, e.to_string()))?
                .into()),
            112 => Ok(device
                .read_model::<Model112>()
                .await
                .map_err(|e| Pv2MqttError::ModelRead(112, e.to_string()))?
                .into()),
            113 => Ok(device
                .read_model::<Model113>()
                .await
                .map_err(|e| Pv2MqttError::ModelRead(113, e.to_string()))?
                .into()),
            _ => Err(Pv2MqttError::UnsupportedModel(model_id)),
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
mod modbus_test_utils;
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
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

    #[tokio::test]
    async fn test_reconnection_logic() {
        // Use tokio::time::pause() to advance time and skip the "real" timeout
        tokio::time::pause();

        let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
        let handle_mock = modbus_test_utils::start_mock_server(addr).await;
        let addr_str = handle_mock.addr.to_string();

        let (tx, _rx) = mpsc::channel(1);
        let token = CancellationToken::new();
        let root_cert_store = Arc::new(rustls::RootCertStore::empty());

        let task = ConnectionTask {
            config: ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: addr_str,
                    tls: false,
                },
                devices: vec![DeviceConfig {
                    unit_id: 1,
                    interval: 10,
                }],
                keep_alive_interval: None,
            },
            mqtt_tx: tx,
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
            token: token.clone(),
            root_cert_store,
        };

        let token_clone = token.clone();
        let task_handle = tokio::spawn(async move { task.run_internal().await });

        // Wait for the first connection
        handle_mock.notify.notified().await;
        assert_eq!(handle_mock.connections.load(Ordering::SeqCst), 1);

        // Advance time. Discovery will fail (no SunSpec marker), but it should NOT reconnect.
        tokio::time::advance(Duration::from_secs(1)).await;

        // Wait some time and check that connections is still 1
        tokio::time::advance(Duration::from_secs(RECONNECT_TIMEOUT_SECS * 2)).await;
        assert_eq!(handle_mock.connections.load(Ordering::SeqCst), 1);

        token_clone.cancel();
        let result = task_handle.await.unwrap();
        assert!(result.is_ok()); // run_internal returns Ok(()) on cancellation
    }

    #[tokio::test]
    async fn test_successful_poll_logic() {
        let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
        let handle_mock = modbus_test_utils::start_mock_server(addr).await;
        let addr_str = handle_mock.addr.to_string();
        let regs = handle_mock.registers;

        // Setup SunSpec discovery registers
        {
            let mut r = regs.lock().unwrap();
            r[40000] = 0x5375; // 'Su'
            r[40001] = 0x6e53; // 'nS'

            // Model 1 (Common)
            r[40002] = 1; // Model ID
            r[40003] = 66; // Length
            // Pre-fill with spaces
            for i in 0..66 {
                r[40004 + i] = 0x2020;
            }
            // Manufacturer (Mn) starts at offset 0 of Model 1 data (r[40004])
            r[40004] = 0x4272; // 'Br'
            r[40005] = 0x616e; // 'an'
            r[40006] = 0x6420; // 'd '
            // Model (Md) starts at offset 16 of Model 1 data (r[40004+16=40020])
            r[40020] = 0x4d6f; // 'Mo'
            r[40021] = 0x6465; // 'de'
            r[40022] = 0x6c20; // 'l '
            // Serial number (SN) starts at offset 50 (40004 + 50 = 40054)
            r[40054] = 0x534e; // 'SN'
            r[40055] = 0x3132; // '12'
            r[40056] = 0x3334; // '34'

            // Model 103 (Three Phase Inverter) at 40070 (40002 + 66 + 2)
            r[40070] = 103; // Model ID
            r[40071] = 50; // Length
            let base = 40072;

            // W at offset 12, W_SF at offset 13
            r[base + 12] = 1000;
            r[base + 13] = 0; // SF = 0

            // St at offset 36
            r[base + 36] = 1; // OFF

            // End of models marker
            r[base + 50] = 0xFFFF;
            r[base + 51] = 0;
        }

        let (tx, mut rx) = mpsc::channel(100);
        let token = CancellationToken::new();
        let root_cert_store = Arc::new(rustls::RootCertStore::empty());

        let task = ConnectionTask {
            config: ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: addr_str,
                    tls: false,
                },
                devices: vec![DeviceConfig {
                    unit_id: 1,
                    interval: 1,
                }], // Short interval
                keep_alive_interval: None,
            },
            mqtt_tx: tx,
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
            token: token.clone(),
            root_cert_store,
        };

        let token_clone = token.clone();
        let task_handle = tokio::spawn(async move { task.run_internal().await });

        // Wait for the mock server to signal a connection
        handle_mock.notify.notified().await;

        // Collect messages
        let mut messages = Vec::new();

        // Wait for all 12 discovery messages with a generous timeout per message
        for _ in 0..12 {
            match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
                Ok(Some(msg)) => messages.push(msg),
                Ok(None) => panic!("Channel closed early (got {})", messages.len()),
                Err(_) => panic!("Timeout waiting for discovery message {}", messages.len()),
            }
        }

        // Wait for next poll (Data + Status)
        // This will naturally wait for the 1s interval to pass in the background task
        for _ in 0..2 {
            match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
                Ok(Some(msg)) => messages.push(msg),
                Ok(None) => panic!("Channel closed early during polling"),
                Err(_) => panic!("Timeout waiting for poll message"),
            }
        }

        assert!(!messages.is_empty(), "Should have received MQTT messages");

        // We expect discovery messages (12) + at least one poll (Data + Status)
        // Discovery messages have topics starting with homeassistant/
        let discovery_msgs: Vec<_> = messages
            .iter()
            .filter(|m| {
                let MqttMessage::Publish { topic, .. } = m;
                topic.starts_with("homeassistant/")
            })
            .collect();
        assert!(
            discovery_msgs.len() >= 12,
            "Should have several discovery messages"
        );

        // Verify one specific discovery message (e.g., Power sensor 'W')
        let w_discovery = discovery_msgs
            .iter()
            .find(|m| {
                let MqttMessage::Publish { topic, .. } = m;
                topic.contains("/W/")
            })
            .expect("Should have discovery for 'W'");

        let MqttMessage::Publish { topic, payload, .. } = w_discovery;
        assert_eq!(*topic, "homeassistant/sensor/SN1234/W/config");
        assert!(payload.contains("\"manufacturer\":\"Brand\""));
        assert!(payload.contains("\"model\":\"Model\""));
        assert!(payload.contains("\"unique_id\":\"solar_SN1234_W\""));

        // Check for data messages
        let data_msgs: Vec<_> = messages
            .iter()
            .filter(|m| {
                let MqttMessage::Publish { topic, .. } = m;
                topic == "solar/inverter/SN1234"
            })
            .collect();
        assert!(!data_msgs.is_empty(), "Should have received poll data");

        let MqttMessage::Publish { payload, .. } = &data_msgs[0];
        let json: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(json["W"], 1000.0);
        assert_eq!(json["St"], "OFF");

        // Check for status messages
        let status_msgs: Vec<_> = messages
            .iter()
            .filter(|m| {
                let MqttMessage::Publish { topic, .. } = m;
                topic == "solar/inverter/SN1234/status"
            })
            .collect();
        assert!(
            !status_msgs.is_empty(),
            "Should have received status updates"
        );

        let MqttMessage::Publish { payload, .. } = &status_msgs[0];
        let json: serde_json::Value = serde_json::from_str(payload).unwrap();
        assert_eq!(json["status"], "OK");

        token_clone.cancel();
        let _ = task_handle.await.unwrap();
    }
}
