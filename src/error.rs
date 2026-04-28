// SPDX-License-Identifier: Apache-2.0

use sunspec::Model;
use sunspec::client::{DiscoveryError, ModbusError as SunSpecModbusError, ReadModelError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModbusError {
    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Protocol error: {0}")]
    Protocol(String),
}

impl ModbusError {
    pub fn is_fatal(&self) -> bool {
        // IO and Timeout are fatal. Protocol errors (like Illegal Function)
        // are usually device-specific and not fatal to the transport.
        match self {
            ModbusError::Timeout(_) | ModbusError::Io(_) => true,
            ModbusError::Protocol(_) => false,
        }
    }
}

#[derive(Error, Debug)]
pub enum Pv2MqttError {
    #[error("Modbus error: {0}")]
    Modbus(#[from] ModbusError),

    #[error("Device discovery failed for unit {0}: {1}")]
    DeviceDiscovery(u8, ModbusError),

    #[error("Model {0} read failed: {1}")]
    ModelRead(u16, ModbusError),

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

    #[error("JSON error: {0}")]
    Json(String),

    #[error("Channel send error: {0}")]
    ChannelSend(String),
}

pub type Result<T> = std::result::Result<T, Pv2MqttError>;

impl From<tokio_modbus::Error> for ModbusError {
    fn from(e: tokio_modbus::Error) -> Self {
        match e {
            tokio_modbus::Error::Transport(io_err) => Self::Io(io_err),
            _ => Self::Protocol(e.to_string()),
        }
    }
}

impl From<SunSpecModbusError> for ModbusError {
    fn from(e: SunSpecModbusError) -> Self {
        match e {
            SunSpecModbusError::Timeout => Self::Timeout("Timeout".to_string()),
            SunSpecModbusError::IO(io_err) => Self::Io(io_err),
            _ => Self::Protocol(e.to_string()),
        }
    }
}

impl From<DiscoveryError> for ModbusError {
    fn from(e: DiscoveryError) -> Self {
        match e {
            DiscoveryError::ModbusError(me) => me.into(),
            _ => Self::Protocol(e.to_string()),
        }
    }
}

impl<M: Model> From<ReadModelError<M>> for ModbusError {
    fn from(e: ReadModelError<M>) -> Self {
        match e {
            ReadModelError::Modbus(me) => me.into(),
            _ => Self::Protocol(e.to_string()),
        }
    }
}

impl From<tokio_modbus::Error> for Pv2MqttError {
    fn from(e: tokio_modbus::Error) -> Self {
        Self::Modbus(ModbusError::from(e))
    }
}

impl From<SunSpecModbusError> for Pv2MqttError {
    fn from(e: SunSpecModbusError) -> Self {
        Self::Modbus(ModbusError::from(e))
    }
}

impl From<DiscoveryError> for Pv2MqttError {
    fn from(e: DiscoveryError) -> Self {
        match e {
            DiscoveryError::ModbusError(me) => Self::Modbus(me.into()),
            _ => Self::Internal(e.to_string()),
        }
    }
}

impl<M: Model> From<ReadModelError<M>> for Pv2MqttError {
    fn from(e: ReadModelError<M>) -> Self {
        match e {
            ReadModelError::Modbus(me) => Self::Modbus(me.into()),
            _ => Self::Internal(e.to_string()),
        }
    }
}

impl From<std::io::Error> for Pv2MqttError {
    fn from(e: std::io::Error) -> Self {
        Self::Modbus(ModbusError::Io(e))
    }
}

impl From<serde_json::Error> for Pv2MqttError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e.to_string())
    }
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Pv2MqttError {
    fn from(e: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Self::ChannelSend(e.to_string())
    }
}

impl From<tokio_serial::Error> for Pv2MqttError {
    fn from(e: tokio_serial::Error) -> Self {
        Self::Modbus(ModbusError::Io(std::io::Error::other(e)))
    }
}

impl Pv2MqttError {
    pub fn category(&self) -> &'static str {
        match self {
            Pv2MqttError::Modbus(e) => match e {
                ModbusError::Timeout(_) => "TIMEOUT_ERROR",
                _ => "CONNECTION_ERROR",
            },
            Pv2MqttError::DeviceDiscovery(_, _) => "DISCOVERY_ERROR",
            Pv2MqttError::ModelRead(_, _) => "READ_ERROR",
            Pv2MqttError::UnsupportedModel(_) => "MODEL_ERROR",
            Pv2MqttError::MqttConnection(_) | Pv2MqttError::MqttPublish(_) => "MQTT_ERROR",
            Pv2MqttError::Config(_) => "CONFIG_ERROR",
            Pv2MqttError::Internal(_) => "INTERNAL_ERROR",
            Pv2MqttError::Json(_) => "JSON_ERROR",
            Pv2MqttError::ChannelSend(_) => "CHANNEL_ERROR",
        }
    }

    /// Returns true if the error is fatal to the current Modbus connection
    /// and should trigger a reconnection.
    pub fn is_fatal(&self) -> bool {
        match self {
            Pv2MqttError::Modbus(e) => e.is_fatal(),
            Pv2MqttError::DeviceDiscovery(_, e) => e.is_fatal(),
            Pv2MqttError::ModelRead(_, e) => e.is_fatal(),
            _ => false,
        }
    }
}
