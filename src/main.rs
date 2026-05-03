// SPDX-License-Identifier: Apache-2.0

mod commands;
mod config;
mod error;
mod homeassistant;
mod modbus;
mod models;
mod mqtt;
mod tasks;
mod tls;

use crate::config::Config;
use crate::error::{Pv2MqttError, Result};
use crate::tasks::TaskManager;
use tracing::{error, info};
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

    // Load native certificates
    let root_cert_store = tls::load_native_certs().await?;

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

    // Initialize and run the task manager
    let manager = TaskManager::new(config, root_cert_store);
    manager.run_until_shutdown().await
}
