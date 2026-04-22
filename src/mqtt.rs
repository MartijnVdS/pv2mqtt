// SPDX-License-Identifier: Apache-2.0

use crate::config::MqttConfig;
use rumqttc::{AsyncClient, Event, Incoming, MqttOptions, QoS, TlsConfiguration, Transport};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{Instrument, debug, error, info, info_span};

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
}

impl MqttTask {
    pub fn new(
        config: MqttConfig,
        rx: mpsc::Receiver<MqttMessage>,
        root_cert_store: Arc<rustls::RootCertStore>,
    ) -> Self {
        Self {
            config,
            rx,
            root_cert_store,
        }
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let span = info_span!("mqtt", client_id = %self.config.client_id);
        self.run_internal().instrument(span).await
    }

    async fn run_internal(mut self) -> anyhow::Result<()> {
        let separator = if self.config.url.contains('?') {
            '&'
        } else {
            '?'
        };
        // Append client_id to the URL. If it already exists in the URL, this
        // appended version takes precedence (last one wins in rumqttc).
        let url_with_client_id = format!(
            "{}{}client_id={}",
            self.config.url,
            separator,
            urlencoding::encode(&self.config.client_id)
        );

        let mut mqttoptions = MqttOptions::parse_url(url_with_client_id)?;
        mqttoptions.set_keep_alive(Duration::from_secs(30));

        if self.config.url.starts_with("mqtts://") || self.config.url.starts_with("ssl://") {
            let client_config = rustls::ClientConfig::builder()
                .with_root_certificates(Arc::clone(&self.root_cert_store))
                .with_no_client_auth();

            mqttoptions.set_transport(Transport::tls_with_config(TlsConfiguration::Rustls(
                Arc::new(client_config),
            )));
        }

        let (client, mut eventloop) = AsyncClient::new(mqttoptions, 20);

        info!("MQTT task starting connection to {}", self.config.url);

        let eventloop_handle = tokio::spawn(
            async move {
                loop {
                    match eventloop.poll().await {
                        Ok(notification) => {
                            debug!("MQTT Notification: {:?}", notification);
                            if let Event::Incoming(Incoming::ConnAck(_)) = notification {
                                info!("MQTT connected");
                            }
                        }
                        Err(e) => {
                            error!("MQTT error: {}", e);
                            tokio::time::sleep(Duration::from_secs(5)).await;
                        }
                    }
                }
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
                    debug!("Publishing to {}: {}", topic, payload);
                    if let Err(e) = client
                        .publish(topic, QoS::AtLeastOnce, retain, payload)
                        .await
                    {
                        error!("Failed to publish MQTT message: {}", e);
                    }
                }
            }
        }

        info!("MQTT task shutting down (channel closed)");
        let _ = client.disconnect().await;
        // Give it a moment to flush the disconnect packet
        tokio::time::sleep(Duration::from_millis(500)).await;
        eventloop_handle.abort();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MqttConfig;

    #[tokio::test]
    async fn test_mqtt_task_shutdown() {
        let (tx, rx) = mpsc::channel(1);
        let config = MqttConfig {
            url: "mqtt://localhost:1883".to_string(),
            client_id: "test".to_string(),
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
        };

        let root_cert_store = Arc::new(rustls::RootCertStore::empty());
        let task = MqttTask::new(config, rx, root_cert_store);
        let handle = tokio::spawn(async move { task.run().await });

        // Drop sender to signal shutdown
        drop(tx);

        // Task should finish cleanly
        let result = tokio::time::timeout(Duration::from_secs(1), handle).await;
        assert!(result.is_ok(), "Task did not shut down in time");
        assert!(result.unwrap().unwrap().is_ok(), "Task returned an error");
    }
}
