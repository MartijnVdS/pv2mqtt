// SPDX-License-Identifier: Apache-2.0

mod config;
mod modbus;
mod models;
mod mqtt;

use crate::config::Config;
use crate::modbus::ConnectionTask;
use crate::mqtt::{MqttMessage, MqttTask};
use anyhow::Context;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    // Install default crypto provider for rustls
    rustls::crypto::ring::default_provider()
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
    .context("Failed to load native certificates")?;

    let root_cert_store = Arc::new(root_cert_store);

    // Load configuration
    let config = match Config::load("pv2mqtt.toml") {
        Ok(c) => {
            info!("Configuration loaded successfully");
            c
        }
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            anyhow::bail!("Configuration error");
        }
    };

    let (mqtt_tx, mqtt_rx) = mpsc::channel::<MqttMessage>(100);
    let shutdown_token = CancellationToken::new();

    // Spawn MQTT task
    let mqtt_config = config.mqtt.clone();
    let mqtt_certs = Arc::clone(&root_cert_store);
    let mqtt_handle = tokio::spawn(async move {
        let task = MqttTask::new(mqtt_config, mqtt_rx, mqtt_certs);
        if let Err(e) = task.run().await {
            error!("MQTT task failed: {}", e);
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

        let handle = tokio::spawn(async move {
            let task = ConnectionTask::new(
                conn_config,
                mqtt_tx,
                topic_prefix,
                ha_prefix,
                token,
                conn_certs,
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
