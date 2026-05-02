// SPDX-License-Identifier: Apache-2.0

mod commands;
mod config;
mod error;
mod modbus;
mod models;
mod mqtt;
mod tls;

use crate::commands::ModbusCommand;
use crate::config::Config;
use crate::error::{Pv2MqttError, Result};
use crate::modbus::ConnectionTask;
use crate::mqtt::{MqttMessage, MqttTask};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

fn get_config_path() -> Result<String> {
    let args: Vec<String> = std::env::args().collect();
    match args.len() {
        1 => Ok("/etc/pv2mqtt.conf".to_string()),
        2 => Ok(args[1].clone()),
        _ => Err(Pv2MqttError::Config(
            "Usage: pv2mqtt [config_file]".to_string(),
        )),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    // Install default crypto provider for rustls
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .ok();

    info!("Starting pv2mqtt");

    // Load native certificates in a blocking task
    let root_cert_store = tokio::task::spawn_blocking(|| {
        let mut root_cert_store = rustls::RootCertStore::empty();
        let certs = rustls_native_certs::load_native_certs();
        for cert in certs.certs {
            if let Err(e) = root_cert_store.add(cert) {
                warn!("Could not add a native certificate: {}", e);
            }
        }
        if !certs.errors.is_empty() {
            warn!(
                "Some native certificates could not be loaded: {:?}",
                certs.errors
            );
        }
        root_cert_store
    })
    .await
    .map_err(|e| Pv2MqttError::Internal(format!("Failed to load native certificates: {}", e)))?;

    let root_cert_store = Arc::new(root_cert_store);

    let config_path = get_config_path()?;
    info!("Loading configuration from {}", config_path);

    // Load configuration
    let config = match Config::load(&config_path) {
        Ok(c) => {
            info!("Configuration loaded successfully");
            c
        }
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            return Err(e);
        }
    };

    let (mqtt_tx, mqtt_rx) = mpsc::channel::<MqttMessage>(100);
    let (cmd_tx, _) = tokio::sync::broadcast::channel::<ModbusCommand>(32);
    let shutdown_token = CancellationToken::new();

    // Spawn MQTT task
    let mqtt_config = config.mqtt.clone();
    let mqtt_certs = Arc::clone(&root_cert_store);
    let mqtt_cmd_tx = cmd_tx.clone();
    let mqtt_handle = tokio::spawn(async move {
        let task = MqttTask::new(mqtt_config, mqtt_rx, mqtt_certs, mqtt_cmd_tx);
        match task.run().await {
            Ok(_) => info!("MQTT task finished cleanly"),
            Err(e) => error!("MQTT task failed: {}", e),
        }
    });

    info!("MQTT task spawned");

    // Spawn Connection tasks
    let mut connection_handles = Vec::new();
    for conn_config in config.connections {
        let mqtt_tx = mqtt_tx.clone();
        let topic_prefix = config.mqtt.topic_prefix.clone();
        let ha_prefix = config.mqtt.ha_prefix.clone();
        let token = shutdown_token.clone();
        let conn_certs = Arc::clone(&root_cert_store);
        let cmd_rx = cmd_tx.subscribe();

        let handle = tokio::spawn(async move {
            let task = ConnectionTask::new(
                conn_config,
                mqtt_tx,
                topic_prefix,
                ha_prefix,
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

    // Wait for shutdown signal (SIGINT (Ctrl+C) or SIGTERM)
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

    // Signal connection tasks to stop
    shutdown_token.cancel();

    // Wait for all connection tasks to finish
    for handle in connection_handles {
        let _ = handle.await;
    }
    info!("All connection tasks finished");

    // Drop the sender to let MQTT task know we are done
    drop(mqtt_tx);

    // Wait for MQTT task to finish
    let _ = mqtt_handle.await;
    info!("MQTT task finished. Goodbye!");

    Ok(())
}
