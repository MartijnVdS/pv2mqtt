use super::InverterConnection;
use crate::config::DeviceConfig;
use crate::models::ActiveControlModel;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct DeviceState<C: InverterConnection> {
    pub config: DeviceConfig,
    pub last_poll: Option<Instant>,
    pub last_success_timestamp: Option<DateTime<Utc>>,
    pub serial: Option<String>,
    pub manufacturer: Option<String>,
    pub model: Option<String>,
    pub version: Option<String>,
    pub supported_model: Option<u16>,
    pub active_control: ActiveControlModel,
    pub device: Option<Arc<C>>,
    pub inverter_topic: Option<Arc<str>>,
    pub status_topic: Option<Arc<str>>,
    pub nameplate_topic: Option<Arc<str>>,
    pub discovery_topic: Option<Arc<str>>,
    pub serialization_buffer: bytes::BytesMut,
}

impl<C: InverterConnection> DeviceState<C> {
    pub fn new(config: DeviceConfig) -> Self {
        Self {
            config,
            last_poll: None,
            last_success_timestamp: None,
            serial: None,
            manufacturer: None,
            model: None,
            version: None,
            supported_model: None,
            active_control: ActiveControlModel::None,
            device: None,
            inverter_topic: None,
            status_topic: None,
            nameplate_topic: None,
            discovery_topic: None,
            serialization_buffer: bytes::BytesMut::with_capacity(1024),
        }
    }

    pub fn clear_connection(&mut self) {
        self.device = None;
    }
}
