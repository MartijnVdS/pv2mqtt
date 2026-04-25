// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum Pv2MqttError {
    #[error("Modbus connection failed: {0}")]
    ModbusConnection(String),

    #[error("Modbus TCP connection failed: {0}")]
    ModbusTcpConnection(String),

    #[error("Modbus timeout: {0}")]
    ModbusTimeout(String),

    #[error("Device discovery failed for unit {0}: {1}")]
    DeviceDiscovery(u8, String),

    #[error("Model {0} read failed: {1}")]
    ModelRead(u16, String),

    #[error("Unsupported SunSpec model: {0}")]
    UnsupportedModel(u16),

    #[error("MQTT connection failed: {0}")]
    MqttConnection(String),

    #[error("MQTT publish failed: {0}")]
    MqttPublish(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("JSON error: {0}")]
    Json(String),

    #[error("Channel send error: {0}")]
    Send(String),
}

pub type Result<T> = std::result::Result<T, Pv2MqttError>;

impl From<tokio_modbus::Error> for Pv2MqttError {
    fn from(e: tokio_modbus::Error) -> Self {
        Self::ModbusConnection(e.to_string())
    }
}

impl From<std::io::Error> for Pv2MqttError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl From<serde_json::Error> for Pv2MqttError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e.to_string())
    }
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Pv2MqttError {
    fn from(e: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Self::Send(e.to_string())
    }
}

impl From<tokio_serial::Error> for Pv2MqttError {
    fn from(e: tokio_serial::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl Pv2MqttError {
    pub fn category(&self) -> &'static str {
        match self {
            Pv2MqttError::ModbusConnection(_) | Pv2MqttError::ModbusTcpConnection(_) => {
                "CONNECTION_ERROR"
            }
            Pv2MqttError::ModbusTimeout(_) => "TIMEOUT_ERROR",
            Pv2MqttError::DeviceDiscovery(_, _) => "DISCOVERY_ERROR",
            Pv2MqttError::ModelRead(_, _) => "READ_ERROR",
            Pv2MqttError::UnsupportedModel(_) => "MODEL_ERROR",
            Pv2MqttError::MqttConnection(_) | Pv2MqttError::MqttPublish(_) => "MQTT_ERROR",
            Pv2MqttError::Config(_) => "CONFIG_ERROR",
            Pv2MqttError::Internal(_) => "INTERNAL_ERROR",
            Pv2MqttError::Io(_) => "IO_ERROR",
            Pv2MqttError::Json(_) => "JSON_ERROR",
            Pv2MqttError::Send(_) => "CHANNEL_ERROR",
        }
    }
}
