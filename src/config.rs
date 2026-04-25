// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::net::SocketAddr;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub mqtt: MqttConfig,
    pub connections: Vec<ConnectionConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MqttConfig {
    pub url: String,
    pub client_id: String,
    pub topic_prefix: String,
    pub ha_prefix: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ConnectionConfig {
    pub name: String,
    pub modbus: ModbusConfig,
    pub devices: Vec<DeviceConfig>,
    pub keep_alive_interval: Option<u64>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Parity {
    None,
    Even,
    Odd,
}

impl Default for Parity {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ModbusConfig {
    #[serde(rename = "tcp")]
    Tcp {
        address: String,
        #[serde(default)]
        tls: bool,
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
}

fn default_polling_interval() -> u64 {
    60
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path).context("Failed to read config file")?;
        let config: Config = toml::from_str(&content).context("Failed to parse config TOML")?;
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
                anyhow::bail!("MQTT {} cannot be empty", name);
            }
            if value.ends_with('/') {
                anyhow::bail!("MQTT {} '{}' cannot end with a slash", name, value);
            }
            if value.contains("//") {
                anyhow::bail!("MQTT {} '{}' cannot contain double slashes", name, value);
            }
        }

        if self.connections.is_empty() {
            anyhow::bail!("At least one connection must be defined");
        }

        for conn in &self.connections {
            if conn.name.trim().is_empty() {
                anyhow::bail!("Connection name cannot be empty or only whitespace");
            }
            if conn.devices.is_empty() {
                anyhow::bail!("Connection '{}' must have at least one device", conn.name);
            }

            match &conn.modbus {
                ModbusConfig::Tcp { address, .. } => {
                    if address.parse::<SocketAddr>().is_err() {
                        // If it's not a direct SocketAddr, check if it's a valid host:port
                        let parts: Vec<&str> = address.split(':').collect();
                        if parts.len() != 2 {
                            anyhow::bail!(
                                "Invalid TCP address '{}' in connection '{}'. Expected 'hostname:port' or 'ip:port'.",
                                address,
                                conn.name
                            );
                        }
                        if parts[0].is_empty() {
                            anyhow::bail!(
                                "Host part of address '{}' cannot be empty in connection '{}'",
                                address,
                                conn.name
                            );
                        }
                        let _: u16 = parts[1].parse().context(format!(
                            "Invalid port in TCP address '{}' in connection '{}'",
                            address, conn.name
                        ))?;
                    }
                }
                ModbusConfig::Rtu { device, .. } => {
                    if device.is_empty() {
                        anyhow::bail!(
                            "RTU device path cannot be empty in connection '{}'",
                            conn.name
                        );
                    }
                }
            }

            let mut unit_ids = std::collections::HashSet::new();
            for device in &conn.devices {
                if !unit_ids.insert(device.unit_id) {
                    anyhow::bail!(
                        "Duplicate unit_id {} in connection '{}'",
                        device.unit_id,
                        conn.name
                    );
                }
                if device.unit_id == 0 {
                    anyhow::bail!("Device unit_id cannot be 0 in connection '{}'", conn.name);
                }
                if device.interval < 1 {
                    anyhow::bail!(
                        "Device polling interval must be at least 1 second in connection '{}'",
                        conn.name
                    );
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
    fn test_strict_prefix_validation() {
        let mut config = Config {
            mqtt: MqttConfig {
                url: "mqtt://localhost:1883".to_string(),
                client_id: "test".to_string(),
                topic_prefix: "solar/".to_string(),
                ha_prefix: "homeassistant".to_string(),
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "127.0.0.1:502".to_string(),
                    tls: false,
                },
                devices: vec![DeviceConfig {
                    unit_id: 1,
                    interval: 10,
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
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "invalid-address".to_string(),
                    tls: false,
                },
                devices: vec![DeviceConfig {
                    unit_id: 1,
                    interval: 10,
                }],
                keep_alive_interval: None,
            }],
        };
        assert!(config.validate().is_err());

        // Should pass with hostname:port
        config.connections[0].modbus = ModbusConfig::Tcp {
            address: "inverter.local:502".to_string(),
            tls: true,
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
            },
            connections: vec![ConnectionConfig {
                name: "test".to_string(),
                modbus: ModbusConfig::Tcp {
                    address: "127.0.0.1:502".to_string(),
                    tls: false,
                },
                devices: vec![
                    DeviceConfig {
                        unit_id: 1,
                        interval: 10,
                    },
                    DeviceConfig {
                        unit_id: 1,
                        interval: 10,
                    },
                ],
                keep_alive_interval: None,
            }],
        };
        // Currently this passes, but it should fail
        assert!(
            config.validate().is_err(),
            "Duplicate unit_ids should be invalid"
        );
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
}
