// SPDX-License-Identifier: Apache-2.0

use crate::commands::{ControlAction, ModbusCommand};
use crate::config::MqttConfig;
use crate::error::{Pv2MqttError, Result};
use rumqttc::{
    AsyncClient, Event, EventLoop, Incoming, MqttOptions, QoS, TlsConfiguration, Transport,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{Instrument, debug, error, info, info_span, trace, warn};

const MQTT_KEEPALIVE_INTERVAL_SECS: u16 = 30;
const MQTT_SHUTDOWN_DELAY: u64 = 500;
const MQTT_MAX_PAYLOAD_SIZE: usize = 64;
const MQTT_RECONNECT_DELAY_SECS: u64 = 5;

#[derive(Debug)]
pub enum MqttMessage {
    Publish {
        topic: String,
        payload: String,
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
    mut eventloop: EventLoop,
    cmd_tx: tokio::sync::broadcast::Sender<ModbusCommand>,
    topic_prefix: String,
) -> Result<()> {
    loop {
        match eventloop.poll().await {
            Ok(notification) => {
                trace!("MQTT Notification: {:?}", notification);
                match notification {
                    Event::Incoming(Incoming::ConnAck(_)) => {
                        info!("MQTT connected");
                    }
                    Event::Incoming(Incoming::Publish(p)) => {
                        handle_incoming_publish(&p, &cmd_tx, &topic_prefix);
                    }
                    _ => {}
                }
            }
            Err(e) => {
                error!("MQTT error: {}", e);
                tokio::time::sleep(Duration::from_secs(MQTT_RECONNECT_DELAY_SECS)).await;
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
    let parts: Vec<&str> = topic.split('/').collect();
    if parts.len() < 5 || parts[0] != topic_prefix || parts[1] != "inverter" || parts[3] != "set" {
        return;
    }

    let serial = parts[2].to_string();
    let register = parts[4];

    let action = match register {
        "Conn" => match payload {
            "true" | "1" | "ON" => Some(ControlAction::Conn(true)),
            "false" | "0" | "OFF" => Some(ControlAction::Conn(false)),
            _ => None,
        },
        "WMaxLimPct" => payload.parse::<f32>().ok().map(ControlAction::WMaxLimPct),
        "WMaxLim_Ena" => match payload {
            "true" | "1" | "ON" => Some(ControlAction::WMaxLimEna(true)),
            "false" | "0" | "OFF" => Some(ControlAction::WMaxLimEna(false)),
            _ => None,
        },
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

    pub async fn run(self) -> Result<()> {
        let span = info_span!("mqtt", client_id = %self.config.client_id);
        self.run_internal().instrument(span).await
    }

    async fn run_internal(mut self) -> Result<()> {
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

        let (client, eventloop) = AsyncClient::builder(mqttoptions).capacity(20).build_async();

        // Subscribe to set topics: {prefix}/inverter/+/set/+
        let subscribe_topic = format!("{}/inverter/+/set/+", self.config.topic_prefix);
        info!("Subscribing to {}", subscribe_topic);
        client
            .subscribe(subscribe_topic, QoS::AtLeastOnce)
            .await
            .map_err(|e| {
                Pv2MqttError::MqttConnection(format!("Failed to subscribe to set topics: {}", e))
            })?;

        info!(
            "MQTT task starting connection to {}",
            self.config.masked_url()
        );

        let topic_prefix = self.config.topic_prefix.clone();
        let eventloop_cmd_tx = self.cmd_tx.clone();
        let eventloop_handle = tokio::spawn(
            async move { run_eventloop(eventloop, eventloop_cmd_tx, topic_prefix).await }
                .instrument(info_span!("mqtt_eventloop")),
        );

        while let Some(msg) = self.rx.recv().await {
            match msg {
                MqttMessage::Publish {
                    topic,
                    payload,
                    retain,
                } => {
                    debug!("Publishing to {}: {}", topic, payload);
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
        // Give it a moment to flush the disconnect packet
        tokio::time::sleep(Duration::from_millis(MQTT_SHUTDOWN_DELAY)).await;
        eventloop_handle.abort();

        Ok(())
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

        // Task should finish cleanly
        let result = tokio::time::timeout(Duration::from_secs(1), handle).await;
        assert!(result.is_ok(), "Task did not shut down in time");
        assert!(result.unwrap().unwrap().is_ok(), "Task returned an error");
    }

    #[tokio::test]
    async fn test_mqtt_task_tls_initialization() {
        let (_tx, rx) = mpsc::channel(1);
        let config = MqttConfig {
            url: "mqtts://localhost:8883".to_string(),
            client_id: "test".to_string(),
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
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
}
