// SPDX-License-Identifier: Apache-2.0

use crate::commands::ModbusCommand;
use crate::config::Config;
use crate::error::Result;
use crate::modbus::{ConnectionTask, SunSpecInverter};
use crate::mqtt::{MqttMessage, MqttTask};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

pub struct TaskManager {
    config: Config,
    root_cert_store: Arc<rustls::RootCertStore>,
    mqtt_tx: mpsc::Sender<MqttMessage>,
    mqtt_rx: Option<mpsc::Receiver<MqttMessage>>,
    cmd_tx: broadcast::Sender<ModbusCommand>,
    shutdown_token: CancellationToken,
}

impl TaskManager {
    pub fn new(config: Config, root_cert_store: Arc<rustls::RootCertStore>) -> Self {
        let (mqtt_tx, mqtt_rx) = mpsc::channel::<MqttMessage>(100);
        let (cmd_tx, _) = broadcast::channel::<ModbusCommand>(32);
        let shutdown_token = CancellationToken::new();

        Self {
            config,
            root_cert_store,
            mqtt_tx,
            mqtt_rx: Some(mqtt_rx),
            cmd_tx,
            shutdown_token,
        }
    }

    pub async fn run_until_shutdown(mut self) -> Result<()> {
        let mqtt_rx = self
            .mqtt_rx
            .take()
            .expect("TaskManager can only be run once");

        // Spawn MQTT task
        let mqtt_handle = self.spawn_mqtt_task(mqtt_rx);
        info!("MQTT task spawned");

        // Spawn Connection tasks
        let connection_handles = self.spawn_connection_tasks();

        // Wait for shutdown signal
        self.wait_for_signal().await?;

        // Graceful shutdown sequence
        info!("Initiating graceful shutdown...");
        self.shutdown_token.cancel();

        // Wait for all connection tasks to finish
        for handle in connection_handles {
            let _ = handle.await;
        }
        info!("All connection tasks finished");

        // Drop the sender to let MQTT task know we are done
        drop(self.mqtt_tx);

        // Wait for MQTT task to finish
        let _ = mqtt_handle.await;
        info!("MQTT task finished. Goodbye!");

        Ok(())
    }

    fn spawn_mqtt_task(&self, mqtt_rx: mpsc::Receiver<MqttMessage>) -> JoinHandle<()> {
        let mqtt_config = self.config.mqtt.clone();
        let mqtt_certs = Arc::clone(&self.root_cert_store);
        let mqtt_cmd_tx = self.cmd_tx.clone();

        tokio::spawn(async move {
            let task = MqttTask::new(mqtt_config, mqtt_rx, mqtt_certs, mqtt_cmd_tx);
            match task.run().await {
                Ok(_) => info!("MQTT task finished cleanly"),
                Err(e) => error!("MQTT task failed: {}", e),
            }
        })
    }

    fn spawn_connection_tasks(&self) -> Vec<JoinHandle<()>> {
        let mut connection_handles = Vec::new();
        for conn_config in self.config.connections.clone() {
            let mqtt_tx = self.mqtt_tx.clone();
            let mqtt_config = self.config.mqtt.clone();
            let token = self.shutdown_token.clone();
            let conn_certs = Arc::clone(&self.root_cert_store);
            let cmd_rx = self.cmd_tx.subscribe();

            let handle: JoinHandle<()> = tokio::spawn(async move {
                let task = ConnectionTask::<SunSpecInverter>::new(
                    conn_config,
                    mqtt_tx,
                    &mqtt_config,
                    token,
                    conn_certs,
                    cmd_rx,
                );

                if let Err(e) = task.run().await {
                    error!("Connection task failed: {}", e);
                }
            });
            connection_handles.push(handle);
        }
        connection_handles
    }

    async fn wait_for_signal(&self) -> Result<()> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sigint = signal(SignalKind::interrupt())?;
            let mut sigterm = signal(SignalKind::terminate())?;

            tokio::select! {
                _ = sigint.recv() => info!("SIGINT received"),
                _ = sigterm.recv() => info!("SIGTERM received"),
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c().await?;
            info!("Shutdown signal received");
        }
        Ok(())
    }
}
