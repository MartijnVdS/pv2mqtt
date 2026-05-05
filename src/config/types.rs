// SPDX-License-Identifier: Apache-2.0

use crate::error::Result;
use serde::Deserialize;
use std::env;
use std::fs;
use tracing::warn;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub mqtt: MqttConfig,
    pub connections: Vec<ConnectionConfig>,
}

#[derive(Deserialize, Clone)]
pub struct MqttConfig {
    pub url: String,
    pub client_id: String,
    pub topic_prefix: String,
    pub ha_prefix: String,
    #[serde(default = "default_true")]
    pub ha_enabled: bool,
    pub ca_path: Option<String>,
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
}

fn default_true() -> bool {
    true
}

impl MqttConfig {
    // Injects MQTT credentials from environment variables or files into the URL.
    // This is called during configuration loading to support secure deployments.
    pub fn inject_env_credentials(&mut self) -> Result<()> {
        self.inject_env_credentials_internal(|key| env::var(key).ok())
    }

    // Internal implementation of credential injection.
    // Uses a closure for variable resolution to allow unit testing without
    // using `unsafe` functions like `std::env::set_var`.
    pub(crate) fn inject_env_credentials_internal<F>(&mut self, get_var: F) -> Result<()>
    where
        F: Fn(&str) -> Option<String>,
    {
        let resolve_secret = |file_var: &str, direct_var: &str| -> Option<String> {
            if let Some(file_path) = get_var(file_var) {
                match fs::read_to_string(&file_path) {
                    Ok(content) => Some(content.trim().to_string()),
                    Err(_) => {
                        warn!("{} is set but the file could not be read", file_var);
                        None
                    }
                }
            } else {
                get_var(direct_var)
            }
        };

        let env_user = resolve_secret("MQTT_USERNAME_FILE", "MQTT_USERNAME");
        let env_pass = resolve_secret("MQTT_PASSWORD_FILE", "MQTT_PASSWORD");

        if env_user.is_some() || env_pass.is_some() {
            if let Ok(mut url) = url::Url::parse(&self.url) {
                if (env_user.is_some() && !url.username().is_empty())
                    || (env_pass.is_some() && url.password().is_some())
                {
                    warn!(
                        "MQTT credentials in environment variables override those configured in the URL"
                    );
                }

                if let Some(user) = env_user {
                    let _ = url.set_username(&user);
                }
                if let Some(pass) = env_pass {
                    let _ = url.set_password(Some(&pass));
                }

                self.url = url.to_string();
            } else {
                warn!(
                    "MQTT URL could not be parsed while trying to inject environment credentials"
                );
            }
        }
        Ok(())
    }

    pub fn masked_url(&self) -> String {
        match url::Url::parse(&self.url) {
            Ok(mut url) => {
                if url.password().is_some() {
                    let _ = url.set_password(Some("********"));
                }
                url.to_string()
            }
            Err(_) => self.url.clone(),
        }
    }
}

impl std::fmt::Debug for MqttConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MqttConfig")
            .field("url", &self.masked_url())
            .field("client_id", &self.client_id)
            .field("topic_prefix", &self.topic_prefix)
            .field("ha_prefix", &self.ha_prefix)
            .field("ha_enabled", &self.ha_enabled)
            .field("cert_path", &self.cert_path)
            .field("key_path", &self.key_path)
            .finish()
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ConnectionConfig {
    pub name: String,
    pub modbus: ModbusConfig,
    pub devices: Vec<DeviceConfig>,
    pub keep_alive_interval: Option<u64>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Parity {
    #[default]
    None,
    Even,
    Odd,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ModbusConfig {
    #[serde(rename = "tcp")]
    Tcp {
        address: String,
        #[serde(default)]
        tls: bool,
        ca_path: Option<String>,
        cert_path: Option<String>,
        key_path: Option<String>,
    },
    #[serde(rename = "rtu")]
    Rtu {
        device: String,
        baud_rate: u32,
        #[serde(default)]
        parity: Parity,
    },
}

#[derive(Debug, Deserialize, Clone)]
pub struct DeviceConfig {
    pub unit_id: u8,
    #[serde(default = "default_polling_interval")]
    pub interval: u64,
    #[serde(default)]
    pub enable_controls: bool,
    pub preferred_model: Option<u16>,
}

fn default_polling_interval() -> u64 {
    60
}
