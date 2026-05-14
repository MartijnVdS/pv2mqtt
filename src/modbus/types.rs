use crate::config::DeviceConfig;
use crate::models::ActiveControlModel;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use std::time::Instant;
use sunspec::client::AsyncDevice;
use tokio::sync::Mutex;
use tokio_modbus::client::Context as ModbusContext;

pub struct DeviceState {
    pub config: DeviceConfig,
    pub last_poll: Option<Instant>,
    pub last_success_timestamp: Option<DateTime<Utc>>,
    pub serial: Option<String>,
    pub manufacturer: Option<String>,
    pub model: Option<String>,
    pub version: Option<String>,
    pub supported_model: Option<u16>,
    pub active_control: ActiveControlModel,
    pub device: Option<AsyncDevice<Arc<Mutex<ModbusContext>>>>,
    pub inverter_topic: Option<Arc<str>>,
    pub status_topic: Option<Arc<str>>,
    pub nameplate_topic: Option<Arc<str>>,
    pub discovery_topic: Option<Arc<str>>,
    pub serialization_buffer: bytes::BytesMut,
}

impl DeviceState {
    pub fn clear_connection(&mut self) {
        self.device = None;
    }
}
