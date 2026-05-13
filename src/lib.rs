// SPDX-License-Identifier: Apache-2.0

pub mod commands;
pub mod config;
pub mod error;
pub mod homeassistant;
pub mod modbus;
pub mod models;
pub mod mqtt;
pub mod taskmanager;
pub mod tls;

use crate::config::Config;
use crate::error::{Pv2MqttError, Result};
use crate::taskmanager::TaskManager;
use tracing::{error, info};

pub fn get_config_path() -> Result<String> {
    let args: Vec<String> = std::env::args().collect();
    match args.len() {
        // No command line arguments: default to /etc
        1 => Ok("/etc/pv2mqtt.conf".to_string()),
        // One command line argument: use that as the config file name
        2 => Ok(args[1].clone()),
        _ => Err(Pv2MqttError::Config(
            "Usage: pv2mqtt [config_file]".to_string(),
        )),
    }
}

pub async fn run() -> Result<()> {
    // Load native certificates
    let root_cert_store = tls::load_native_certs().await?;

    let config_path = get_config_path()?;
    info!("Loading configuration from {}", config_path);

    // Load configuration
    let config = match Config::load(&config_path) {
        Ok(cfg) => {
            info!("Configuration loaded successfully");
            info!(
                "Home Assistant Autodiscovery: {}",
                if cfg.mqtt.ha_enabled {
                    "Enabled"
                } else {
                    "Disabled"
                }
            );
            cfg
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
