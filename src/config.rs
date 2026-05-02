// SPDX-License-Identifier: Apache-2.0

use crate::error::{Pv2MqttError, Result};
use serde::Deserialize;
use std::fs;
use std::net::SocketAddr;
use std::path::Path;

const MAX_POLL_SECS: u64 = 3600;
const MAX_KEEPALIVE_SECS: u64 = 3600;
const MAX_MODBUS_UNIT_ID: u8 = 247;

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
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
}

impl MqttConfig {
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
}

fn default_polling_interval() -> u64 {
    60
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref).map_err(|e| {
            Pv2MqttError::Config(format!("Failed to read config file {:?}: {}", path_ref, e))
        })?;
        let config: Config = toml::from_str(&content)
            .map_err(|e| Pv2MqttError::Config(format!("Failed to parse config TOML: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let prefixes = [
            ("topic_prefix", &self.mqtt.topic_prefix),
            ("ha_prefix", &self.mqtt.ha_prefix),
        ];

        for (name, value) in prefixes {
            if value.is_empty() {
                return Err(Pv2MqttError::Config(format!(
                    "MQTT {} cannot be empty",
                    name
                )));
            }
            if value.ends_with('/') {
                return Err(Pv2MqttError::Config(format!(
                    "MQTT {} '{}' cannot end with a slash",
                    name, value
                )));
            }
            if value.contains("//") {
                return Err(Pv2MqttError::Config(format!(
                    "MQTT {} '{}' cannot contain double slashes",
                    name, value
                )));
            }
        }

        if self.connections.is_empty() {
            return Err(Pv2MqttError::Config(
                "At least one connection must be defined".to_string(),
            ));
        }

        for conn in &self.connections {
            if conn.name.trim().is_empty() {
                return Err(Pv2MqttError::Config(
                    "Connection name cannot be empty or only whitespace".to_string(),
                ));
            }
            if conn.devices.is_empty() {
                return Err(Pv2MqttError::Config(format!(
                    "Connection '{}' must have at least one device",
                    conn.name
                )));
            }

            match &conn.modbus {
                ModbusConfig::Tcp { address, .. } => {
                    if address.parse::<SocketAddr>().is_err() {
                        // If it's not a direct SocketAddr, check if it's a valid host:port
                        let parts: Vec<&str> = address.split(':').collect();
                        if parts.len() != 2 {
                            return Err(Pv2MqttError::Config(format!(
                                "Invalid TCP address '{}' in connection '{}'. Expected 'hostname:port' or 'ip:port'.",
                                address, conn.name
                            )));
                        }
                        if parts[0].is_empty() {
                            return Err(Pv2MqttError::Config(format!(
                                "Host part of address '{}' cannot be empty in connection '{}'",
                                address, conn.name
                            )));
                        }
                        let _: u16 = parts[1].parse().map_err(|e| {
                            Pv2MqttError::Config(format!(
                                "Invalid port in TCP address '{}' in connection '{}': {}",
                                address, conn.name, e
                            ))
                        })?;
                    }
                }
                ModbusConfig::Rtu {
                    device, baud_rate, ..
                } => {
                    if device.is_empty() {
                        return Err(Pv2MqttError::Config(format!(
                            "RTU device path cannot be empty in connection '{}'",
                            conn.name
                        )));
                    }
                    if *baud_rate == 0 {
                        return Err(Pv2MqttError::Config(format!(
                            "RTU baud rate cannot be 0 in connection '{}'",
                            conn.name
                        )));
                    }
                }
            }

            if let Some(ka) = conn.keep_alive_interval
                && ka > MAX_KEEPALIVE_SECS
            {
                return Err(Pv2MqttError::Config(format!(
                    "Keep-alive interval {} is too large in connection '{}' (max {}s)",
                    ka, conn.name, MAX_KEEPALIVE_SECS
                )));
            }

            let mut unit_ids = std::collections::HashSet::new();
            for device in &conn.devices {
                if !unit_ids.insert(device.unit_id) {
                    return Err(Pv2MqttError::Config(format!(
                        "Duplicate unit_id {} in connection '{}'",
                        device.unit_id, conn.name
                    )));
                }
                if device.unit_id == 0 || device.unit_id > MAX_MODBUS_UNIT_ID {
                    return Err(Pv2MqttError::Config(format!(
                        "Device unit_id {} is out of valid Modbus range (1-{}) in connection '{}'",
                        device.unit_id, MAX_MODBUS_UNIT_ID, conn.name
                    )));
                }
                if !(1..=MAX_POLL_SECS).contains(&device.interval) {
                    return Err(Pv2MqttError::Config(format!(
                        "Device polling interval {} is out of reasonable range (1-{}s) in connection '{}'",
                        device.interval, MAX_POLL_SECS, conn.name
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mqtt_url_masking() {
        let config = MqttConfig {
            url: "mqtt://user:password@localhost:1883".to_string(),
            client_id: "test".to_string(),
            topic_prefix: "pv2mqtt".to_string(),
            ha_prefix: "homeassistant".to_string(),
            cert_path: None,
            key_path: None,
        };
        assert_eq!(config.masked_url(), "mqtt://user:********@localhost:1883");

        let config_no_pass = MqttConfig {
            url: "mqtt://localhost:1883".to_string(),
            client_id: "test".to_string(),
            topic_prefix: "pv2mqtt".to_string(),
            ha_prefix: "homeassistant".to_string(),
            cert_path: None,
            key_path: None,
        };
        assert_eq!(config_no_pass.masked_url(), "mqtt://localhost:1883");
    }

    #[test]
    fn test_strict_prefix_validation() {
        let mut config = Config {
            mqtt: MqttConfig {
                url: "mqtt://localhost:1883".to_string(),
                client_id: "test".to_string(),
                topic_prefix: "solar/".to_string(),
                ha_prefix: "homeassistant".to_string(),
                cert_path: None,
                key_path: None,
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "127.0.0.1:502".to_string(),
                    tls: false,
                    cert_path: None,
                    key_path: None,
                },
                devices: vec![DeviceConfig {
                    unit_id: 1,
                    interval: 10,
                    enable_controls: false,
                }],
                keep_alive_interval: None,
            }],
        };

        // Should fail because of trailing slash
        assert!(config.validate().is_err());

        // Should fail because of double slash
        config.mqtt.topic_prefix = "solar//data".to_string();
        assert!(config.validate().is_err());

        // Should pass
        config.mqtt.topic_prefix = "solar".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_tcp_address() {
        let mut config = Config {
            mqtt: MqttConfig {
                url: "mqtt://localhost:1883".to_string(),
                client_id: "test".to_string(),
                topic_prefix: "solar".to_string(),
                ha_prefix: "homeassistant".to_string(),
                cert_path: None,
                key_path: None,
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "invalid-address".to_string(),
                    tls: false,
                    cert_path: None,
                    key_path: None,
                },
                devices: vec![DeviceConfig {
                    unit_id: 1,
                    interval: 10,
                    enable_controls: false,
                }],
                keep_alive_interval: None,
            }],
        };
        assert!(config.validate().is_err());

        // Should pass with hostname:port
        config.connections[0].modbus = ModbusConfig::Tcp {
            address: "inverter.local:502".to_string(),
            tls: true,
            cert_path: None,
            key_path: None,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_empty_connections_or_devices() {
        let mut config = Config {
            mqtt: MqttConfig {
                url: "mqtt://localhost:1883".to_string(),
                client_id: "test".to_string(),
                topic_prefix: "solar".to_string(),
                ha_prefix: "homeassistant".to_string(),
                cert_path: None,
                key_path: None,
            },
            connections: vec![],
        };

        // Empty connections
        assert!(config.validate().is_err());

        // Empty devices
        config.connections.push(ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: "127.0.0.1:502".to_string(),
                tls: false,
                cert_path: None,
                key_path: None,
            },

            devices: vec![],
            keep_alive_interval: None,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_duplicate_unit_ids() {
        let config = Config {
            mqtt: MqttConfig {
                url: "mqtt://localhost:1883".to_string(),
                client_id: "test".to_string(),
                topic_prefix: "solar".to_string(),
                ha_prefix: "homeassistant".to_string(),
                cert_path: None,
                key_path: None,
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "127.0.0.1:502".to_string(),
                    tls: false,
                    cert_path: None,
                    key_path: None,
                },
                devices: vec![
                    DeviceConfig {
                        unit_id: 1,
                        interval: 10,
                        enable_controls: false,
                    },
                    DeviceConfig {
                        unit_id: 1,
                        interval: 10,
                        enable_controls: false,
                    },
                ],
                keep_alive_interval: None,
            }],
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_unit_id_range() {
        let mut config = Config {
            mqtt: MqttConfig {
                url: "mqtt://localhost:1883".to_string(),
                client_id: "test".to_string(),
                topic_prefix: "solar".to_string(),
                ha_prefix: "homeassistant".to_string(),
                cert_path: None,
                key_path: None,
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "127.0.0.1:502".to_string(),
                    tls: false,
                    cert_path: None,
                    key_path: None,
                },
                devices: vec![DeviceConfig {
                    unit_id: 248, // Out of range
                    interval: 10,
                    enable_controls: false,
                }],
                keep_alive_interval: None,
            }],
        };
        assert!(config.validate().is_err());

        if let Some(conn) = config.connections.get_mut(0) {
            conn.devices[0].unit_id = 0; // Also out of range
        }
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_interval_range() {
        let mut config = Config {
            mqtt: MqttConfig {
                url: "mqtt://localhost:1883".to_string(),
                client_id: "test".to_string(),
                topic_prefix: "solar".to_string(),
                ha_prefix: "homeassistant".to_string(),
                cert_path: None,
                key_path: None,
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "127.0.0.1:502".to_string(),
                    tls: false,
                    cert_path: None,
                    key_path: None,
                },
                devices: vec![DeviceConfig {
                    unit_id: 1,
                    interval: 4000, // Too large
                    enable_controls: false,
                }],
                keep_alive_interval: None,
            }],
        };
        assert!(config.validate().is_err());

        // Test boundary (minimum)
        if let Some(conn) = config.connections.get_mut(0) {
            conn.devices[0].interval = 1;
        }
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_baud_rate() {
        let config = Config {
            mqtt: MqttConfig {
                url: "mqtt://localhost:1883".to_string(),
                client_id: "test".to_string(),
                topic_prefix: "solar".to_string(),
                ha_prefix: "homeassistant".to_string(),
                cert_path: None,
                key_path: None,
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Rtu {
                    device: "/dev/ttyUSB0".to_string(),
                    baud_rate: 0, // Invalid
                    parity: Parity::None,
                },
                devices: vec![DeviceConfig {
                    unit_id: 1,
                    interval: 10,
                    enable_controls: false,
                }],
                keep_alive_interval: None,
            }],
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_rtu_config_parsing() {
        let content = r#"
            [mqtt]
            url = "mqtt://localhost"
            client_id = "test"
            topic_prefix = "pv2mqtt"
            ha_prefix = "homeassistant"

            [[connections]]
            name = "rtu_test"
            [connections.modbus]
            type = "rtu"
            device = "/dev/ttyUSB0"
            baud_rate = 9600
            parity = "even"
            [[connections.devices]]
            unit_id = 1
            interval = 60
        "#;

        let config: Config = toml::from_str(content).unwrap();
        let conn = &config.connections[0];
        if let ModbusConfig::Rtu {
            device,
            baud_rate,
            parity,
        } = &conn.modbus
        {
            assert_eq!(device, "/dev/ttyUSB0");
            assert_eq!(*baud_rate, 9600);
            assert_eq!(*parity, Parity::Even);
        } else {
            panic!("Expected RTU config");
        }
    }

    #[test]
    fn test_rtu_config_default_parity() {
        let content = r#"
            [mqtt]
            url = "mqtt://localhost"
            client_id = "test"
            topic_prefix = "pv2mqtt"
            ha_prefix = "homeassistant"

            [[connections]]
            name = "rtu_test"
            [connections.modbus]
            type = "rtu"
            device = "/dev/ttyUSB0"
            baud_rate = 9600
            [[connections.devices]]
            unit_id = 1
            interval = 60
        "#;

        let config: Config = toml::from_str(content).unwrap();
        let conn = &config.connections[0];
        if let ModbusConfig::Rtu { parity, .. } = &conn.modbus {
            assert_eq!(*parity, Parity::None);
        } else {
            panic!("Expected RTU config");
        }
    }

    #[test]
    fn test_config_load_success() {
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("pv2mqtt_test_valid.toml");
        let content = r#"
            [mqtt]
            url = "mqtt://localhost:1883"
            client_id = "pv2mqtt_test"
            topic_prefix = "solar"
            ha_prefix = "homeassistant"

            [[connections]]
            name = "inverter1"
            keep_alive_interval = 30
            [connections.modbus]
            type = "tcp"
            address = "127.0.0.1:502"
            tls = false
            [[connections.devices]]
            unit_id = 1
            interval = 10
        "#;
        fs::write(&config_path, content).unwrap();

        let result = Config::load(&config_path);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.mqtt.client_id, "pv2mqtt_test");
        assert_eq!(config.connections.len(), 1);
        assert_eq!(config.connections[0].name, "inverter1");

        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn test_config_load_file_not_found() {
        let result = Config::load("non_existent_config_file.toml");
        assert!(result.is_err());
        match result.unwrap_err() {
            Pv2MqttError::Config(msg) => assert!(msg.contains("Failed to read config file")),
            e => panic!("Expected Config error, got {:?}", e),
        }
    }

    #[test]
    fn test_config_load_invalid_toml() {
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("pv2mqtt_test_invalid.toml");
        let content = "this is not TOML";
        fs::write(&config_path, content).unwrap();

        let result = Config::load(&config_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            Pv2MqttError::Config(msg) => assert!(msg.contains("Failed to parse config TOML")),
            e => panic!("Expected Config error, got {:?}", e),
        }

        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn test_config_load_validation_failure() {
        let temp_dir = std::env::temp_dir();
        let config_path = temp_dir.join("pv2mqtt_test_validation_fail.toml");
        let content = r#"
            [mqtt]
            url = "mqtt://localhost:1883"
            client_id = "test"
            topic_prefix = "solar/" # Invalid trailing slash
            ha_prefix = "homeassistant"

            [[connections]]
            name = "test"
            [connections.modbus]
            type = "tcp"
            address = "127.0.0.1:502"
            [[connections.devices]]
            unit_id = 1
        "#;
        fs::write(&config_path, content).unwrap();

        let result = Config::load(&config_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            Pv2MqttError::Config(msg) => assert!(msg.contains("cannot end with a slash")),
            e => panic!("Expected Config error, got {:?}", e),
        }

        let _ = fs::remove_file(config_path);
    }
}
