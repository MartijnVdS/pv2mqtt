// SPDX-License-Identifier: Apache-2.0

use crate::commands::{ControlAction, ModbusCommand};
use crate::config::MqttConfig;
use crate::error::{Pv2MqttError, Result};
use rumqttc::{
    AsyncClient, Event, EventLoop, Incoming, MqttOptions, Outgoing, QoS, TlsConfiguration,
    Transport,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, error, info, info_span, trace, warn};

const MQTT_KEEPALIVE_INTERVAL_SECS: u16 = 30;
const MQTT_MAX_PAYLOAD_SIZE: usize = 64;
const MQTT_RECONNECT_DELAY_SECS: u64 = 5;
const MQTT_SHUTDOWN_TIMEOUT_MILLIS: u64 = 500;

#[derive(Debug)]
pub enum MqttMessage {
    Publish {
        topic: String,
        payload: bytes::Bytes,
        retain: bool,
    },
}

pub struct MqttTask {
    config: MqttConfig,
    rx: mpsc::Receiver<MqttMessage>,
    root_cert_store: Arc<rustls::RootCertStore>,
    cmd_tx: tokio::sync::broadcast::Sender<ModbusCommand>,
}

async fn run_eventloop(
    client: AsyncClient,
    mut eventloop: EventLoop,
    cmd_tx: tokio::sync::broadcast::Sender<ModbusCommand>,
    topic_prefix: String,
    shutdown_token: CancellationToken,
) -> Result<()> {
    let subscribe_topic = format!("{}/inverter/+/set/+", topic_prefix);
    loop {
        tokio::select! {
            biased;
            _ = shutdown_token.cancelled() => {
                debug!("MQTT eventloop shutting down due to signal");
                return Ok(());
            }
            res = eventloop.poll() => {
                match res {
                    Ok(notification) => {
                        trace!("MQTT Notification: {:?}", notification);
                        match notification {
                            Event::Incoming(Incoming::ConnAck(_)) => {
                                info!("MQTT connected");
                                debug!("Re-subscribing to {}", subscribe_topic);
                                if let Err(e) = client.subscribe(&subscribe_topic, QoS::AtLeastOnce).await {
                                    error!("Failed to subscribe to {}: {}", subscribe_topic, e);
                                }
                            }
                            Event::Incoming(Incoming::Publish(p)) => {
                                handle_incoming_publish(&p, &cmd_tx, &topic_prefix);
                            }
                            Event::Outgoing(Outgoing::Disconnect) => {
                                debug!("MQTT disconnect sent, exiting eventloop");
                                return Ok(());
                            }
                            _ => {}
                        }
                    }
                    Err(e) => {
                        error!("MQTT error: {}", e);
                        tokio::select! {
                            _ = shutdown_token.cancelled() => return Ok(()),
                            _ = tokio::time::sleep(Duration::from_secs(MQTT_RECONNECT_DELAY_SECS)) => {}
                        }
                    }
                }
            }
        }
    }
}

fn handle_incoming_publish(
    p: &rumqttc::Publish,
    cmd_tx: &tokio::sync::broadcast::Sender<ModbusCommand>,
    topic_prefix: &str,
) {
    let topic = match std::str::from_utf8(&p.topic) {
        Ok(s) => s,
        Err(_) => {
            warn!("Rejected MQTT message with non-UTF8 topic");
            return;
        }
    };

    // DoS protection: check payload size
    if p.payload.len() > MQTT_MAX_PAYLOAD_SIZE {
        warn!(
            "Rejected MQTT message on {} with excessive payload size: {} bytes",
            topic,
            p.payload.len()
        );
        return;
    }

    let payload = match std::str::from_utf8(&p.payload) {
        Ok(s) => s.trim(),
        Err(_) => {
            warn!("Rejected MQTT message on {} with invalid UTF-8", topic);
            return;
        }
    };

    // Expected topic: {prefix}/inverter/{serial}/set/{register}
    if !topic.starts_with(topic_prefix) {
        return;
    }

    let rest = &topic[topic_prefix.len()..];
    if !rest.starts_with("/inverter/") {
        return;
    }

    let mut parts = rest[10..].split('/'); // skip "/inverter/"
    let (Some(serial_part), Some(p_set), Some(register)) =
        (parts.next(), parts.next(), parts.next())
    else {
        warn!("Received message for an unknown MQTT topic: {}", topic);
        return;
    };

    if p_set != "set" || parts.next().is_some() {
        warn!(
            "Received message for an unknown or malformed MQTT topic: {}",
            topic
        );
        return;
    }

    let serial = serial_part.to_string();

    let action = match register {
        "Conn" => parse_mqtt_bool(payload).map(ControlAction::Conn),
        "WMaxLimPct" => payload.parse::<f32>().ok().map(ControlAction::WMaxLimPct),
        "WMaxLim_Ena" => parse_mqtt_bool(payload).map(ControlAction::WMaxLimEna),
        _ => None,
    };

    if let Some(action) = action {
        info!("Broadcasting command for {}: {:?}", serial, action);
        let _ = cmd_tx.send(ModbusCommand { serial, action });
    } else {
        warn!(
            "Received unknown or invalid command for {}: {}",
            register, payload
        );
    }
}

fn parse_mqtt_bool(payload: &str) -> Option<bool> {
    if payload.eq_ignore_ascii_case("true") || payload == "1" || payload.eq_ignore_ascii_case("on")
    {
        Some(true)
    } else if payload.eq_ignore_ascii_case("false")
        || payload == "0"
        || payload.eq_ignore_ascii_case("off")
    {
        Some(false)
    } else {
        None
    }
}

impl MqttTask {
    pub fn new(
        config: MqttConfig,
        rx: mpsc::Receiver<MqttMessage>,
        root_cert_store: Arc<rustls::RootCertStore>,
        cmd_tx: tokio::sync::broadcast::Sender<ModbusCommand>,
    ) -> Self {
        Self {
            config,
            rx,
            root_cert_store,
            cmd_tx,
        }
    }

    #[tracing::instrument(name="mqtt", skip(self), fields(client_id=self.config.client_id))]
    pub async fn run(mut self) -> Result<()> {
        let mqttoptions = self.configure_mqtt_options()?;
        let (client, eventloop) = AsyncClient::builder(mqttoptions).capacity(20).build();

        info!(
            "MQTT task starting connection to {}",
            self.config.masked_url()
        );

        let topic_prefix = self.config.topic_prefix.clone();
        let eventloop_cmd_tx = self.cmd_tx.clone();
        let client_clone = client.clone();
        let shutdown_token = CancellationToken::new();
        let shutdown_token_clone = shutdown_token.clone();

        let eventloop_handle = tokio::spawn(
            async move {
                run_eventloop(
                    client_clone,
                    eventloop,
                    eventloop_cmd_tx,
                    topic_prefix,
                    shutdown_token_clone,
                )
                .await
            }
            .instrument(info_span!("mqtt_eventloop")),
        );

        while let Some(msg) = self.rx.recv().await {
            match msg {
                MqttMessage::Publish {
                    topic,
                    payload,
                    retain,
                } => {
                    debug!(
                        "Publishing to {}: {}",
                        topic,
                        String::from_utf8_lossy(&payload)
                    );
                    if let Err(e) = client
                        .publish(topic, QoS::AtLeastOnce, retain, payload)
                        .await
                    {
                        let pv_err = Pv2MqttError::MqttPublish(e.to_string());
                        error!("{}", pv_err);
                    }
                }
            }
        }

        info!("MQTT task shutting down (channel closed)");
        let _ = client.disconnect().await;

        // Signal the event loop to stop if it's currently in a retry sleep.
        // The event loop will still finish processing the disconnect packet if possible
        // because the poll() branch in its select! will compete with the cancellation.
        shutdown_token.cancel();

        // Wait for eventloop to finish or timeout as fallback
        let _ = tokio::time::timeout(
            Duration::from_millis(MQTT_SHUTDOWN_TIMEOUT_MILLIS),
            eventloop_handle,
        )
        .await;

        Ok(())
    }

    fn configure_mqtt_options(&self) -> Result<MqttOptions> {
        let mut url = url::Url::parse(&self.config.url).map_err(|e| {
            Pv2MqttError::MqttConnection(format!("Failed to parse MQTT URL: {}", e))
        })?;
        url.query_pairs_mut()
            .append_pair("client_id", &self.config.client_id);

        let scheme = url.scheme().to_owned();
        let is_tls = scheme == "mqtts" || scheme == "ssl";

        // MqttOptions::parse_url in rumqttc-next handles mqtts:// by requiring
        // a default TLS configuration. Since we want to provide our own
        // via set_transport, we change the scheme to mqtt:// for parsing.
        if is_tls {
            url.set_scheme("mqtt").map_err(|_| {
                Pv2MqttError::MqttConnection("failed to set scheme to mqtt".to_string())
            })?;
        }

        let mut mqttoptions = MqttOptions::parse_url(url.as_str()).map_err(|e| {
            Pv2MqttError::MqttConnection(format!("Failed to parse URL for rumqttc: {}", e))
        })?;
        mqttoptions.set_keep_alive(MQTT_KEEPALIVE_INTERVAL_SECS);

        if is_tls {
            if self.config.cert_path.is_some() || self.config.key_path.is_some() {
                info!("Using mTLS for MQTT connection");
            }

            let client_config = crate::tls::create_client_config(
                Arc::clone(&self.root_cert_store),
                self.config.ca_path.as_deref(),
                self.config.cert_path.as_deref(),
                self.config.key_path.as_deref(),
            )?;

            mqttoptions.set_transport(Transport::tls_with_config(TlsConfiguration::Rustls(
                Arc::new(client_config),
            )));
        }

        Ok(mqttoptions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MqttConfig;
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_mqtt_task_shutdown() {
        let (tx, rx) = mpsc::channel(1);
        let config = MqttConfig {
            url: "mqtt://localhost:1883".to_string(),
            client_id: "test".to_string(),
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        };

        let root_cert_store = Arc::new(rustls::RootCertStore::empty());
        let (cmd_tx, _) = tokio::sync::broadcast::channel(1);
        let task = MqttTask::new(config, rx, root_cert_store, cmd_tx);
        let handle = tokio::spawn(async move { task.run().await });

        // Drop sender to signal shutdown
        drop(tx);

        // Task should finish cleanly and fast (no more real-world sleep)
        let result = tokio::time::timeout(Duration::from_secs(1), handle).await;
        assert!(result.is_ok(), "Task did not shut down in time");
        assert!(result.unwrap().unwrap().is_ok(), "Task returned an error");
    }

    #[tokio::test]
    #[traced_test]
    async fn test_mqtt_task_tls_initialization() {
        let (_tx, rx) = mpsc::channel(1);
        let config = MqttConfig {
            url: "mqtts://localhost:8883".to_string(),
            client_id: "test".to_string(),
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        };

        let root_cert_store = Arc::new(rustls::RootCertStore::empty());
        let (cmd_tx, _) = tokio::sync::broadcast::channel(1);
        let task = MqttTask::new(config, rx, root_cert_store, cmd_tx);

        // Don't run the MQTT task here because it would try to connect
        // to localhost:8883 and fail/hang.

        // But we should at least check if the masked_url is correct.
        assert_eq!(task.config.masked_url(), "mqtts://localhost:8883");
    }

    #[test]
    fn test_handle_incoming_publish_multilevel_prefix() {
        let (cmd_tx, mut cmd_rx) = tokio::sync::broadcast::channel(1);
        let prefix = "home/solar";

        // 1. Valid multi-level prefix
        let p = rumqttc::Publish {
            topic: "home/solar/inverter/SN123/set/WMaxLimPct".into(),
            payload: "75.5".into(),
            dup: false,
            retain: false,
            qos: QoS::AtLeastOnce,
            pkid: 0,
            properties: None,
        };
        handle_incoming_publish(&p, &cmd_tx, prefix);
        let cmd = cmd_rx.try_recv().expect("Should have received a command");
        assert_eq!(cmd.serial, "SN123");
        assert!(matches!(cmd.action, ControlAction::WMaxLimPct(val) if val == 75.5));

        // 2. Invalid prefix
        let p = rumqttc::Publish {
            topic: "other/solar/inverter/SN123/set/WMaxLimPct".into(),
            payload: "75.5".into(),
            dup: false,
            retain: false,
            qos: QoS::AtLeastOnce,
            pkid: 0,
            properties: None,
        };
        handle_incoming_publish(&p, &cmd_tx, prefix);
        assert!(cmd_rx.try_recv().is_err());

        // 3. Malformed path
        let p = rumqttc::Publish {
            topic: "home/solar/inverter/SN123/something/WMaxLimPct".into(),
            payload: "75.5".into(),
            dup: false,
            retain: false,
            qos: QoS::AtLeastOnce,
            pkid: 0,
            properties: None,
        };
        handle_incoming_publish(&p, &cmd_tx, prefix);
        assert!(cmd_rx.try_recv().is_err());
    }
}
