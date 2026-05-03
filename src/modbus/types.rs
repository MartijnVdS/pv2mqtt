use crate::config::DeviceConfig;
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
    pub device: Option<AsyncDevice<Arc<Mutex<ModbusContext>>>>,
}

impl DeviceState {
    pub fn clear_connection(&mut self) {
        self.device = None;
    }
}

pub struct DiscoveryContext<'a> {
    pub manufacturer: &'a str,
    pub model: &'a str,
    pub version: Option<&'a str>,
    pub name: &'a str,
    pub value_path: Option<String>,
    pub unit: Option<&'a str>,
    pub device_class: Option<&'a str>,
    pub state_class: Option<&'a str>,
    pub label: &'a str,
    pub enabled_by_default: bool,
    pub options: Option<Vec<&'static str>>,
    pub component: Option<&'static str>,
    pub command_topic: Option<String>,
}
