// SPDX-License-Identifier: Apache-2.0

use super::types::{Config, ModbusConfig};
use crate::error::{Pv2MqttError, Result};
use std::net::SocketAddr;

pub const MAX_POLL_SECS: u64 = 3600;
pub const MAX_KEEPALIVE_SECS: u64 = 3600;
pub const MAX_MODBUS_UNIT_ID: u8 = 247;

impl Config {
    pub fn validate(&self) -> Result<()> {
        self.validate_mqtt_prefixes()?;

        if self.connections.is_empty() {
            return Err(Pv2MqttError::Config(
                "At least one connection must be defined".to_string(),
            ));
        }

        for conn in &self.connections {
            conn.validate()?;
        }

        Ok(())
    }

    fn validate_mqtt_prefixes(&self) -> Result<()> {
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
        Ok(())
    }
}

use super::types::ConnectionConfig;

impl ConnectionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Pv2MqttError::Config(
                "Connection name cannot be empty or only whitespace".to_string(),
            ));
        }
        if self.devices.is_empty() {
            return Err(Pv2MqttError::Config(format!(
                "Connection '{}' must have at least one device",
                self.name
            )));
        }

        self.validate_modbus_config()?;

        if let Some(ka) = self.keep_alive_interval
            && ka > MAX_KEEPALIVE_SECS
        {
            return Err(Pv2MqttError::Config(format!(
                "Keep-alive interval {} is too large in connection '{}' (max {}s)",
                ka, self.name, MAX_KEEPALIVE_SECS
            )));
        }

        self.validate_devices()?;

        Ok(())
    }

    fn validate_modbus_config(&self) -> Result<()> {
        match &self.modbus {
            ModbusConfig::Tcp { address, .. } => {
                if address.parse::<SocketAddr>().is_err() {
                    // If it's not a direct SocketAddr, check if it's a valid host:port
                    if let Some(colon_idx) = address.rfind(':') {
                        let host = &address[..colon_idx];
                        let port_str = &address[colon_idx + 1..];

                        if host.is_empty() {
                            return Err(Pv2MqttError::Config(format!(
                                "Host part of address '{}' cannot be empty in connection '{}'",
                                address, self.name
                            )));
                        }

                        // Heuristic: If there's more than one colon, it's likely an IPv6 address.
                        // IPv6 addresses with ports MUST be bracketed in this config format
                        // to avoid ambiguity (e.g. [::1]:502).
                        if host.contains(':') && !host.starts_with('[') {
                            return Err(Pv2MqttError::Config(format!(
                                "Ambiguous TCP address '{}' in connection '{}'. IPv6 addresses with ports must be bracketed, e.g., '[::1]:502'.",
                                address, self.name
                            )));
                        }

                        port_str.parse::<u16>().map_err(|e| {
                            Pv2MqttError::Config(format!(
                                "Invalid port in TCP address '{}' in connection '{}': {}",
                                address, self.name, e
                            ))
                        })?;
                    } else {
                        return Err(Pv2MqttError::Config(format!(
                            "Invalid TCP address '{}' in connection '{}'. Expected 'hostname:port' or '[ipv6]:port'.",
                            address, self.name
                        )));
                    }
                }
            }
            ModbusConfig::Rtu {
                device, baud_rate, ..
            } => {
                if device.is_empty() {
                    return Err(Pv2MqttError::Config(format!(
                        "RTU device path cannot be empty in connection '{}'",
                        self.name
                    )));
                }
                if *baud_rate == 0 {
                    return Err(Pv2MqttError::Config(format!(
                        "RTU baud rate cannot be 0 in connection '{}'",
                        self.name
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_devices(&self) -> Result<()> {
        let mut unit_ids = std::collections::HashSet::new();
        for device in &self.devices {
            if !unit_ids.insert(device.unit_id) {
                return Err(Pv2MqttError::Config(format!(
                    "Duplicate unit_id {} in connection '{}'",
                    device.unit_id, self.name
                )));
            }
            if device.unit_id == 0 || device.unit_id > MAX_MODBUS_UNIT_ID {
                return Err(Pv2MqttError::Config(format!(
                    "Device unit_id {} is out of valid Modbus range (1-{}) in connection '{}'",
                    device.unit_id, MAX_MODBUS_UNIT_ID, self.name
                )));
            }
            if device.interval < 1 || device.interval > MAX_POLL_SECS {
                return Err(Pv2MqttError::Config(format!(
                    "Device polling interval {} is out of reasonable range (1-{}s) in connection '{}'",
                    device.interval, MAX_POLL_SECS, self.name
                )));
            }
        }
        Ok(())
    }
}
