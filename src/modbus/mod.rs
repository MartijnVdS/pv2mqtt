// SPDX-License-Identifier: Apache-2.0

mod command;
mod discovery;
mod polling;
mod types;

pub use types::DeviceState;

use crate::config::{ConnectionConfig, ModbusConfig, Parity};
use crate::error::{ModbusError, Pv2MqttError, Result};
use crate::homeassistant::HomeAssistantIntegration;
use crate::mqtt::MqttMessage;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sunspec::client::{AsyncClient, Config as SunSpecConfig};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};
use tokio::sync::{Mutex, mpsc};
use tokio_modbus::{
    Slave,
    client::{Context as ModbusContext, rtu, tcp},
};
use tokio_rustls::{TlsConnector, rustls::pki_types::ServerName};
use tokio_serial::SerialStream;
use tokio_util::{either, sync::CancellationToken};
use tracing::{Instrument, debug, error, info, info_span, warn};

pub const COMMAND_POLL_DELAY_MILLIS: u64 = 250;
pub const CONNECT_TIMEOUT_SECS: u64 = 10;
pub const DEFAULT_IDLE_SLEEP_SECS: u64 = 3600;
pub const MIN_SLEEP_MILLIS: u64 = 10;
pub const POLL_TIMEOUT_SECS: u64 = 10;
pub const READ_TIMEOUT_SECS: u64 = 10;
pub const RECONNECT_TIMEOUT_SECS: u64 = 10;
pub const STALE_DATA_WAIT_MILLIS: u64 = 100;

// SunSpec Model 123 (Immediate Controls) Data-Relative Offsets (Spec Offset - 2)
// These match the 'offset' attribute in SunSpec smdx_00123.xml
pub const M123_CONN_OFFSET: u16 = 2;
pub const M123_WMAX_LIM_PCT_OFFSET: u16 = 3;
pub const M123_WMAX_LIM_ENA_OFFSET: u16 = 7;
pub const M123_WMAX_LIM_PCT_SF_OFFSET: u16 = 21;

// SunSpec Model 704 (DER AC Controls) Data-Relative Offsets (Spec Offset - 2)
// These match the 'offset' attribute in the SunSpec specification
pub const M704_WMAX_LIM_ENA_OFFSET: u16 = 12;
pub const M704_WMAX_LIM_PCT_OFFSET: u16 = 13;
pub const M704_WMAX_LIM_PCT_SF_OFFSET: u16 = 52;

pub struct ConnectionTask {
    pub config: ConnectionConfig,
    pub mqtt_tx: mpsc::Sender<MqttMessage>,
    pub ha: HomeAssistantIntegration,
    pub ha_enabled: bool,
    pub token: CancellationToken,
    pub root_cert_store: Arc<rustls::RootCertStore>,
    pub cmd_rx: tokio::sync::broadcast::Receiver<crate::commands::ModbusCommand>,
}

impl ConnectionTask {
    pub fn new(
        config: ConnectionConfig,
        mqtt_tx: mpsc::Sender<MqttMessage>,
        mqtt_config: &crate::config::MqttConfig,
        token: CancellationToken,
        root_cert_store: Arc<rustls::RootCertStore>,
        cmd_rx: tokio::sync::broadcast::Receiver<crate::commands::ModbusCommand>,
    ) -> Self {
        Self {
            config,
            mqtt_tx,
            ha: HomeAssistantIntegration::new(
                mqtt_config.topic_prefix.clone(),
                mqtt_config.ha_prefix.clone(),
            ),
            ha_enabled: mqtt_config.ha_enabled,
            token,
            root_cert_store,
            cmd_rx,
        }
    }

    pub async fn run(self) -> Result<()> {
        let span = info_span!("connection", name = %self.config.name);
        self.run_internal().instrument(span).await
    }

    pub async fn run_internal(self) -> Result<()> {
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
                active_control: crate::models::ActiveControlModel::None,
                device: None,
                inverter_topic: None,
                status_topic: None,
            })
            .collect();

        let mut first_run = true;

        while !self.token.is_cancelled() {
            for device_state in devices.iter_mut() {
                device_state.clear_connection();
                device_state.last_poll = None;
            }

            if !first_run {
                info!("Attempting to reconnect");
            }

            let result: Result<()> = async {
                let ctx = self.establish_connection().await?;
                let ctx = Arc::new(Mutex::new(ctx));

                self.run_polling_loop(ctx, &mut devices, self.cmd_rx.resubscribe())
                    .await?;
                Ok(())
            }
            .await;

            if let Err(e) = result {
                error!(
                    "Connection error: {}. Reconnecting in {}s...",
                    e, RECONNECT_TIMEOUT_SECS
                );

                for device_state in devices.iter_mut() {
                    device_state.clear_connection();
                }

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
        ctx: Arc<Mutex<ModbusContext>>,
        devices: &mut [DeviceState],
        mut cmd_rx: tokio::sync::broadcast::Receiver<crate::commands::ModbusCommand>,
    ) -> Result<()> {
        let client = AsyncClient::new(
            ctx.clone(),
            SunSpecConfig {
                read_timeout: None,
                ..SunSpecConfig::default()
            },
        );

        let mut last_activity = Instant::now();
        let keep_alive_interval = self.config.keep_alive_interval.unwrap_or(30);
        let keep_alive_duration = Duration::from_secs(keep_alive_interval);

        loop {
            let now = Instant::now();
            let mut next_wakeup =
                if keep_alive_interval > 0 && devices.iter().any(|d| d.device.is_some()) {
                    last_activity + keep_alive_duration
                } else {
                    now + Duration::from_secs(DEFAULT_IDLE_SLEEP_SECS)
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
                .max(Duration::from_millis(MIN_SLEEP_MILLIS));

            tokio::select! {
                biased;
                _ = self.token.cancelled() => {
                    info!("Shutting down polling loop");
                    return Ok(());
                }
                res = cmd_rx.recv() => {
                    match res {
                        Ok(cmd) => {
                            self.handle_command(&ctx, devices, cmd).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Command channel lagged by {} messages", n);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            debug!("Command channel closed");
                        }
                    }
                    // No continue here; we fall through to check if any polls are due.
                    // This prevents a flood of commands from starving the polling logic.
                }
                _ = tokio::time::sleep(sleep_duration) => {}
            }

            let now = Instant::now();

            if keep_alive_interval > 0
                && now.duration_since(last_activity) >= keep_alive_duration
                && let Some((state, device)) = devices
                    .iter()
                    .find_map(|d| d.device.as_ref().map(|dev| (d, dev)))
            {
                let ping_span = info_span!("keep_alive", unit_id = state.config.unit_id);
                if let Err(e) = self.ping_device(device).instrument(ping_span).await {
                    error!("Keep-alive failed: {}", e);
                    return Err(e);
                }
                last_activity = Instant::now();
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
                    device_state.last_poll = Some(now);
                    let res = if device_state.device.is_none() {
                        match self.discover_device(&client, device_state).await {
                            Ok(_) => self.perform_device_poll(device_state, now).await,
                            Err(e) => Err(e),
                        }
                    } else {
                        self.perform_device_poll(device_state, now).await
                    };

                    if let Err(e) = res {
                        if e.is_fatal() {
                            return Err(e);
                        }
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

    // Set up TLS on a TCP connection
    async fn connect_tls(
        &self,
        stream: tokio::net::TcpStream,
        address: &String,
        ca_path: Option<&str>,
        cert_path: Option<&str>,
        key_path: Option<&str>,
    ) -> Result<tokio_rustls::client::TlsStream<tokio::net::TcpStream>> {
        let client_config = crate::tls::create_client_config(
            Arc::clone(&self.root_cert_store),
            ca_path,
            cert_path,
            key_path,
        )?;
        let connector = TlsConnector::from(Arc::new(client_config));
        let host = address
            .split(':')
            .next()
            .ok_or_else(|| Pv2MqttError::Config("Invalid address".to_string()))?;
        let server_name = ServerName::try_from(host)
            .map_err(|e| Pv2MqttError::Config(format!("invalid server name: {}", e)))?
            .to_owned();
        let tls_stream = tokio::time::timeout(
            Duration::from_secs(CONNECT_TIMEOUT_SECS),
            connector.connect(server_name, stream),
        )
        .await
        .map_err(|_| {
            ModbusError::Timeout(format!("Timeout during TLS handshake with {}", address))
        })??;

        Ok(tls_stream)
    }

    async fn drain_stale_data<T>(&self, stream: &mut T) -> Result<()>
    where
        T: AsyncRead + AsyncWrite + Send + Unpin,
    {
        let mut discard_buf = [0u8; 512];
        let mut total_discarded = 0;

        while let Ok(Ok(n)) = tokio::time::timeout(
            Duration::from_millis(STALE_DATA_WAIT_MILLIS),
            stream.read(&mut discard_buf),
        )
        .await
        {
            total_discarded += n;
            if n == 0 {
                break;
            }
        }

        if total_discarded > 0 {
            info!(
                "Discarded {} stale bytes from previous session",
                total_discarded
            );
        }

        Ok(())
    }

    async fn establish_connection(&self) -> Result<ModbusContext> {
        match &self.config.modbus {
            ModbusConfig::Tcp {
                address,
                tls,
                ca_path,
                cert_path,
                key_path,
            } => {
                info!("Connecting to Modbus TCP at {} (TLS: {})", address, tls);
                let tcp_stream = tokio::time::timeout(
                    Duration::from_secs(CONNECT_TIMEOUT_SECS),
                    tokio::net::TcpStream::connect(address),
                )
                .await
                .map_err(|_| {
                    ModbusError::Timeout(format!("Timeout connecting to {}", address))
                })??;

                tcp_stream.set_nodelay(true)?;

                let mut stream = if *tls {
                    let tls_stream = self
                        .connect_tls(
                            tcp_stream,
                            address,
                            ca_path.as_deref(),
                            cert_path.as_deref(),
                            key_path.as_deref(),
                        )
                        .await?;
                    either::Either::Left(tls_stream)
                } else {
                    either::Either::Right(tcp_stream)
                };

                self.drain_stale_data(&mut stream).await?;

                Ok(tcp::attach(stream))
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
                let port = SerialStream::open(&builder)
                    .map_err(|e| ModbusError::Io(std::io::Error::other(e)))?;
                Ok(rtu::attach_slave(port, Slave(0)))
            }
        }
    }
}

#[cfg(test)]
mod tests;
