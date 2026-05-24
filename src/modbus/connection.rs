// SPDX-License-Identifier: Apache-2.0

use crate::config::DeviceConfig;
use crate::error::{ModbusError, Pv2MqttError, Result};
use crate::modbus::{POLL_TIMEOUT_SECS, READ_TIMEOUT_SECS};
use crate::models::{
    ActiveControlModel, InverterData, NameplateData, SUPPORTED_INVERTER_DATA_MODELS, apply_sf,
    poll_and_apply,
};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use sunspec::client::{AsyncClient, AsyncDevice, AsyncModbusClient, Config as SunSpecConfig};
use sunspec::models::model1::Model1;
use tokio::sync::Mutex;
use tokio_modbus::client::Context as ModbusContext;
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct DeviceMetadata {
    pub serial: String,
    pub manufacturer: String,
    pub model: String,
    pub version: Option<String>,
    pub supported_model: u16,
    pub active_control: ActiveControlModel,
}

pub trait InverterConnection: Send + Sync + 'static {
    fn slave_id(&self) -> u8;

    fn discover(
        ctx: Arc<Mutex<ModbusContext>>,
        unit_id: u8,
        config: &DeviceConfig,
    ) -> impl Future<Output = Result<Self>> + Send
    where
        Self: Sized;

    fn read_metadata(&self) -> impl Future<Output = Result<DeviceMetadata>> + Send;
    fn read_nameplate(
        &self,
        available_models: &[u16],
    ) -> impl Future<Output = Result<Option<NameplateData>>> + Send;
    fn poll(
        &self,
        model_id: u16,
        data: &mut InverterData,
    ) -> impl Future<Output = Result<()>> + Send;
    fn ping(&self) -> impl Future<Output = Result<()>> + Send;
    fn write_registers(&self, addr: u16, data: &[u16]) -> impl Future<Output = Result<()>> + Send;
    fn read_registers(
        &self,
        addr: u16,
        count: u16,
    ) -> impl Future<Output = Result<Vec<u16>>> + Send;
    fn supported_model_ids(&self) -> Vec<u16>;
    fn read_model<M: sunspec::Model + Send + Sync>(&self)
    -> impl Future<Output = Result<M>> + Send;
}

pub struct SunSpecInverter {
    device: AsyncDevice<Arc<Mutex<ModbusContext>>>,
    supported_model: u16,
    active_control: ActiveControlModel,
}

impl SunSpecInverter {
    pub fn new(
        device: AsyncDevice<Arc<Mutex<ModbusContext>>>,
        supported_model: u16,
        active_control: ActiveControlModel,
    ) -> Self {
        Self {
            device,
            supported_model,
            active_control,
        }
    }

    async fn read_model_internal<M>(
        device: &AsyncDevice<Arc<Mutex<ModbusContext>>>,
        unit_id: u8,
    ) -> Result<M>
    where
        M: sunspec::Model + Send + Sync,
    {
        let res = tokio::time::timeout(
            Duration::from_secs(POLL_TIMEOUT_SECS),
            device.read_model::<M>(),
        )
        .await;

        match res {
            Ok(Ok(m)) => Ok(m),
            Ok(Err(e)) => Err(Pv2MqttError::DeviceDiscovery(unit_id, ModbusError::from(e))),
            Err(_) => Err(Pv2MqttError::Modbus(ModbusError::Timeout(format!(
                "Timeout reading Model {} for device {}",
                M::ID,
                unit_id
            )))),
        }
    }

    fn identify_inverter_model(
        unit_id: u8,
        available_models: &[u16],
        preferred: Option<u16>,
    ) -> Result<u16> {
        let mut selected_model = None;

        if let Some(preferred_id) = preferred {
            if available_models.contains(&preferred_id) {
                info!(
                    "Using preferred SunSpec model {} for unit {}",
                    preferred_id, unit_id
                );
                selected_model = Some(preferred_id);
            } else {
                warn!(
                    "Preferred SunSpec model {} is not supported by hardware for unit {}. Falling back to default priority list.",
                    preferred_id, unit_id
                );
            }
        }

        if selected_model.is_none() {
            selected_model = SUPPORTED_INVERTER_DATA_MODELS
                .iter()
                .find(|&&id| available_models.contains(&id))
                .copied();
        }

        match selected_model {
            Some(model_id) => Ok(model_id),
            None => {
                let available = available_models
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(Pv2MqttError::Internal(format!(
                    "No supported inverter model found for unit {}. Available: {}",
                    unit_id, available
                )))
            }
        }
    }

    fn identify_control_model(
        available_models: &[u16],
        enable_controls: bool,
        device: &AsyncDevice<Arc<Mutex<ModbusContext>>>,
    ) -> ActiveControlModel {
        if !enable_controls {
            return ActiveControlModel::None;
        }

        if available_models.contains(&704) {
            info!("Using SunSpec Model 704 for controls on unit");
            ActiveControlModel::Model704 {
                base_addr: device.models.m704.addr,
            }
        } else if available_models.contains(&123) {
            info!("Using SunSpec Model 123 for controls on unit");
            ActiveControlModel::Model123 {
                base_addr: device.models.m123.addr,
            }
        } else {
            warn!(
                "Device has controls enabled in config, but neither Model 704 nor Model 123 is supported by hardware."
            );
            ActiveControlModel::None
        }
    }
}

impl InverterConnection for SunSpecInverter {
    fn slave_id(&self) -> u8 {
        self.device.slave_id
    }

    fn supported_model_ids(&self) -> Vec<u16> {
        self.device.models.supported_model_ids()
    }

    async fn discover(
        ctx: Arc<Mutex<ModbusContext>>,
        unit_id: u8,
        config: &DeviceConfig,
    ) -> Result<Self> {
        let client = AsyncClient::new(
            ctx.clone(),
            SunSpecConfig {
                read_timeout: None,
                ..SunSpecConfig::default()
            },
        );

        // Note: We do NOT call set_slave here manually. The SunSpec AsyncClient
        // and its AsyncModbusClient implementation for Arc<Mutex<Context>>
        // handle setting the slave ID atomically before every request.
        let device_res = tokio::time::timeout(
            Duration::from_secs(POLL_TIMEOUT_SECS),
            client.device(unit_id),
        )
        .await;

        let device = match device_res {
            Ok(Ok(d)) => {
                debug!("Successfully identified unit {} as SunSpec device", unit_id);
                d
            }
            Ok(Err(e)) => {
                warn!("Modbus error during discovery for unit {}: {}", unit_id, e);
                return Err(Pv2MqttError::DeviceDiscovery(unit_id, ModbusError::from(e)));
            }
            Err(_) => {
                warn!("Timeout during discovery for unit {}", unit_id);
                return Err(Pv2MqttError::Modbus(ModbusError::Timeout(format!(
                    "Timeout discovering device {}",
                    unit_id
                ))));
            }
        };

        let available_models = device.models.supported_model_ids();
        let supported_model =
            Self::identify_inverter_model(unit_id, &available_models, config.preferred_model)?;

        let active_control =
            Self::identify_control_model(&available_models, config.enable_controls, &device);

        Ok(Self {
            device,
            supported_model,
            active_control,
        })
    }

    async fn read_metadata(&self) -> Result<DeviceMetadata> {
        let m1 = Self::read_model_internal::<Model1>(&self.device, self.device.slave_id).await?;
        Ok(DeviceMetadata {
            serial: m1.sn.trim().to_string(),
            manufacturer: m1.mn.trim().to_string(),
            model: m1.md.trim().to_string(),
            version: m1
                .vr
                .as_ref()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty()),
            supported_model: self.supported_model,
            active_control: self.active_control,
        })
    }

    async fn read_nameplate(&self, available_models: &[u16]) -> Result<Option<NameplateData>> {
        if available_models.contains(&702) {
            if let Ok(m702) = Self::read_model_internal::<sunspec::models::model702::Model702>(
                &self.device,
                self.device.slave_id,
            )
            .await
            {
                return Ok(Some(NameplateData {
                    w_max: apply_sf(m702.w_max_rtg, m702.w_sf.unwrap_or(0)),
                    va_max: apply_sf(m702.va_max_rtg, m702.va_sf.unwrap_or(0)),
                    var_max_inj: apply_sf(m702.var_max_inj_rtg, m702.var_sf.unwrap_or(0)),
                    var_max_abs: apply_sf(m702.var_max_abs_rtg, m702.var_sf.unwrap_or(0)),
                }));
            }
        } else if available_models.contains(&120)
            && let Ok(m120) = Self::read_model_internal::<sunspec::models::model120::Model120>(
                &self.device,
                self.device.slave_id,
            )
            .await
        {
            return Ok(Some(NameplateData {
                w_max: apply_sf(m120.w_rtg, m120.w_rtg_sf),
                va_max: apply_sf(m120.va_rtg, m120.va_rtg_sf),
                var_max_inj: apply_sf(m120.v_ar_rtg_q1 as u16, m120.v_ar_rtg_sf),
                var_max_abs: apply_sf(m120.v_ar_rtg_q4 as u16, m120.v_ar_rtg_sf),
            }));
        }
        Ok(None)
    }

    async fn poll(&self, model_id: u16, data: &mut InverterData) -> Result<()> {
        poll_and_apply(model_id, self, data).await
    }

    async fn ping(&self) -> Result<()> {
        debug!("Sending keep-alive");
        let addr = self.device.models.m1.addr;
        let _regs = tokio::time::timeout(
            Duration::from_secs(READ_TIMEOUT_SECS),
            self.device
                .client
                .read_registers(self.device.slave_id, addr, 2),
        )
        .await
        .map_err(|_| {
            Pv2MqttError::Internal(format!(
                "Keep-alive timeout for unit {}",
                self.device.slave_id
            ))
        })??;

        Ok(())
    }

    async fn write_registers(&self, addr: u16, data: &[u16]) -> Result<()> {
        tokio::time::timeout(
            Duration::from_secs(POLL_TIMEOUT_SECS),
            self.device
                .client
                .write_registers(self.device.slave_id, addr, data),
        )
        .await
        .map_err(|_| {
            Pv2MqttError::Modbus(ModbusError::Timeout(format!(
                "Timeout writing registers for unit {}",
                self.device.slave_id
            )))
        })??;
        Ok(())
    }

    async fn read_registers(&self, addr: u16, count: u16) -> Result<Vec<u16>> {
        let regs = tokio::time::timeout(
            Duration::from_secs(POLL_TIMEOUT_SECS),
            self.device
                .client
                .read_registers(self.device.slave_id, addr, count),
        )
        .await
        .map_err(|_| {
            Pv2MqttError::Modbus(ModbusError::Timeout(format!(
                "Timeout reading registers for unit {}",
                self.device.slave_id
            )))
        })??;
        Ok(regs)
    }

    async fn read_model<M: sunspec::Model + Send + Sync>(&self) -> Result<M> {
        let res = tokio::time::timeout(
            Duration::from_secs(POLL_TIMEOUT_SECS),
            self.device.read_model::<M>(),
        )
        .await;

        match res {
            Ok(Ok(m)) => Ok(m),
            Ok(Err(e)) => Err(e.into()),
            Err(_) => Err(Pv2MqttError::Modbus(ModbusError::Timeout(format!(
                "Timeout reading Model {} for unit {}",
                M::ID,
                self.device.slave_id
            )))),
        }
    }
}

#[cfg(test)]
#[derive(Clone)]
pub struct MockInverter {
    pub slave_id: u8,
    pub metadata: DeviceMetadata,
    pub nameplate: Option<NameplateData>,
    pub model_ids: Vec<u16>,
}

#[cfg(test)]
impl InverterConnection for MockInverter {
    fn slave_id(&self) -> u8 {
        self.slave_id
    }
    async fn discover(
        _ctx: Arc<Mutex<ModbusContext>>,
        unit_id: u8,
        config: &DeviceConfig,
    ) -> Result<Self> {
        Ok(Self {
            slave_id: unit_id,
            metadata: DeviceMetadata {
                serial: "MOCK_SERIAL".to_string(),
                manufacturer: "MockMFG".to_string(),
                model: "MockMDL".to_string(),
                version: Some("1.0".to_string()),
                supported_model: config.preferred_model.unwrap_or(103),
                active_control: ActiveControlModel::None,
            },
            nameplate: None,
            model_ids: vec![1, 103],
        })
    }
    async fn read_metadata(&self) -> Result<DeviceMetadata> {
        Ok(self.metadata.clone())
    }
    async fn read_nameplate(&self, _available_models: &[u16]) -> Result<Option<NameplateData>> {
        Ok(self.nameplate.clone())
    }
    async fn poll(&self, _model_id: u16, _data: &mut InverterData) -> Result<()> {
        Ok(())
    }
    async fn ping(&self) -> Result<()> {
        Ok(())
    }
    async fn write_registers(&self, _addr: u16, _data: &[u16]) -> Result<()> {
        Ok(())
    }
    async fn read_registers(&self, _addr: u16, _count: u16) -> Result<Vec<u16>> {
        Ok(vec![0])
    }
    fn supported_model_ids(&self) -> Vec<u16> {
        self.model_ids.clone()
    }
    async fn read_model<M: sunspec::Model + Send + Sync>(&self) -> Result<M> {
        Err(Pv2MqttError::Internal(
            "MockInverter::read_model not implemented".to_string(),
        ))
    }
}
