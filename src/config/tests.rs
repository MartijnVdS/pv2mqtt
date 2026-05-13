// SPDX-License-Identifier: Apache-2.0

use super::types::*;
use crate::error::Pv2MqttError;
use std::fs;

#[test]
fn test_mqtt_url_masking() {
    let config = MqttConfig {
        url: "mqtt://user:pass@localhost:1883".to_string(),
        client_id: "pv2mqtt".to_string(),
        topic_prefix: "pv2mqtt".to_string(),
        ha_prefix: "homeassistant".to_string(),
        ha_enabled: true,

        ca_path: None,
        cert_path: None,
        key_path: None,
    };
    assert_eq!(config.masked_url(), "mqtt://user:********@localhost:1883");

    let config_no_pass = MqttConfig {
        url: "mqtt://localhost:1883".to_string(),
        client_id: "test".to_string(),
        topic_prefix: "pv2mqtt".to_string(),
        ha_prefix: "homeassistant".to_string(),
        ha_enabled: true,
        ca_path: None,
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
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        },
        connections: vec![ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: "127.0.0.1:502".to_string(),
                tls: false,
                ca_path: Option::None,
                cert_path: Option::None,
                key_path: Option::None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 10,
                enable_controls: false,
                preferred_model: None,
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
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        },
        connections: vec![ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: "invalid-address".to_string(),
                tls: false,
                ca_path: Option::None,
                cert_path: Option::None,
                key_path: Option::None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 10,
                enable_controls: false,
                preferred_model: None,
            }],
            keep_alive_interval: None,
        }],
    };
    assert!(config.validate().is_err());

    // Should pass with hostname:port
    config.connections[0].modbus = ModbusConfig::Tcp {
        address: "inverter.local:502".to_string(),
        tls: true,
        ca_path: Option::None,
        cert_path: Option::None,
        key_path: Option::None,
    };
    assert!(config.validate().is_ok());

    // Should pass with IPv6 address
    config.connections[0].modbus = ModbusConfig::Tcp {
        address: "[::1]:502".to_string(),
        tls: false,
        ca_path: None,
        cert_path: None,
        key_path: None,
    };
    assert!(config.validate().is_ok());

    // Should fail with invalid IPv6 (missing port)
    config.connections[0].modbus = ModbusConfig::Tcp {
        address: "::1".to_string(),
        tls: false,
        ca_path: None,
        cert_path: None,
        key_path: None,
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_empty_connections_or_devices() {
    let mut config = Config {
        mqtt: MqttConfig {
            url: "mqtt://localhost:1883".to_string(),
            client_id: "test".to_string(),
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
            ha_enabled: true,
            ca_path: None,
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
            ca_path: Option::None,
            cert_path: Option::None,
            key_path: Option::None,
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
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        },
        connections: vec![ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: "127.0.0.1:502".to_string(),
                tls: false,
                ca_path: Option::None,
                cert_path: Option::None,
                key_path: Option::None,
            },
            devices: vec![
                DeviceConfig {
                    unit_id: 1,
                    interval: 10,
                    enable_controls: false,
                    preferred_model: None,
                },
                DeviceConfig {
                    unit_id: 1,
                    interval: 10,
                    enable_controls: false,
                    preferred_model: None,
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
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        },
        connections: vec![ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: "127.0.0.1:502".to_string(),
                tls: false,
                ca_path: Option::None,
                cert_path: Option::None,
                key_path: Option::None,
            },
            devices: vec![DeviceConfig {
                unit_id: 248, // Out of range
                interval: 10,
                enable_controls: false,
                preferred_model: None,
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
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        },
        connections: vec![ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: "127.0.0.1:502".to_string(),
                tls: false,
                ca_path: Option::None,
                cert_path: Option::None,
                key_path: Option::None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 4000, // Too large
                enable_controls: false,
                preferred_model: None,
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
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        },
        connections: vec![ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Rtu {
                device: "/dev/ttyUSB0".to_string(),
                baud_rate: 0, // Invalid
                parity: Parity::None,
                stop_bits: 1,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 10,
                enable_controls: false,
                preferred_model: None,
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
        stop_bits,
    } = &conn.modbus
    {
        assert_eq!(device, "/dev/ttyUSB0");
        assert_eq!(*baud_rate, 9600);
        assert_eq!(*parity, Parity::Even);
        assert_eq!(*stop_bits, 1)
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

#[test]
fn test_mqtt_env_and_file_credentials() {
    let mut mqtt = MqttConfig {
        url: "mqtt://localhost:1883".to_string(),
        client_id: "test".to_string(),
        topic_prefix: "pv2mqtt".to_string(),
        ha_prefix: "homeassistant".to_string(),
        ha_enabled: true,
        ca_path: None,
        cert_path: None,
        key_path: None,
    };

    // 1. Test basic environment variables
    mqtt.inject_env_credentials_internal(|key| match key {
        "MQTT_USERNAME" => Some("env_user".to_string()),
        "MQTT_PASSWORD" => Some("env_pass".to_string()),
        _ => None,
    })
    .unwrap();
    assert_eq!(mqtt.url, "mqtt://env_user:env_pass@localhost:1883");

    // Prepare files for file-based tests
    let temp_dir = std::env::temp_dir();
    let user_file = temp_dir.join("mqtt_user.txt");
    let pass_file = temp_dir.join("mqtt_pass.txt");
    fs::write(&user_file, "file_user\n").unwrap();
    fs::write(&pass_file, "file_pass  ").unwrap();
    let user_file_str = user_file.to_str().unwrap().to_string();
    let pass_file_str = pass_file.to_str().unwrap().to_string();

    // 2. Test file-based credentials
    mqtt.url = "mqtt://localhost:1883".to_string();
    mqtt.inject_env_credentials_internal(|key| match key {
        "MQTT_USERNAME_FILE" => Some(user_file_str.clone()),
        "MQTT_PASSWORD_FILE" => Some(pass_file_str.clone()),
        _ => None,
    })
    .unwrap();
    assert_eq!(mqtt.url, "mqtt://file_user:file_pass@localhost:1883");

    // 3. Test file-based credentials taking precedence over regular variables
    mqtt.url = "mqtt://localhost:1883".to_string();
    mqtt.inject_env_credentials_internal(|key| match key {
        "MQTT_USERNAME" => Some("env_user".to_string()),
        "MQTT_PASSWORD" => Some("env_pass".to_string()),
        "MQTT_USERNAME_FILE" => Some(user_file_str.clone()),
        "MQTT_PASSWORD_FILE" => Some(pass_file_str.clone()),
        _ => None,
    })
    .unwrap();
    assert_eq!(mqtt.url, "mqtt://file_user:file_pass@localhost:1883");

    // Cleanup
    let _ = fs::remove_file(user_file);
    let _ = fs::remove_file(pass_file);
}
