// SPDX-License-Identifier: Apache-2.0

pub mod types;
pub mod validation;

#[cfg(test)]
mod tests;

pub use types::*;

use crate::error::{Pv2MqttError, Result};
use std::fs;
use std::path::Path;

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref).map_err(|e| {
            Pv2MqttError::Config(format!("Failed to read config file {:?}: {}", path_ref, e))
        })?;
        let mut config: Config = toml::from_str(&content)
            .map_err(|e| Pv2MqttError::Config(format!("Failed to parse config TOML: {}", e)))?;
        config.mqtt.inject_env_credentials()?;
        config.validate()?;
        Ok(config)
    }
}
