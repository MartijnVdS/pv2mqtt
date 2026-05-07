use super::{ConnectionTask, DeviceState, POLL_TIMEOUT_SECS};
use crate::error::{ModbusError, Pv2MqttError, Result};
use crate::models::{ActiveControlModel, SUPPORTED_MODELS};
use std::sync::Arc;
use std::time::Duration;
use sunspec::client::AsyncClient;
use sunspec::models::model1::Model1;
use tokio::sync::Mutex;
use tokio_modbus::Slave;
use tokio_modbus::client::Context as ModbusContext;
use tokio_modbus::slave::SlaveContext;
use tracing::{Instrument, debug, info, info_span, warn};

impl ConnectionTask {
    pub async fn discover_device(
        &self,
        client: &AsyncClient<Arc<Mutex<ModbusContext>>>,
        device_state: &mut DeviceState,
    ) -> Result<()> {
        let unit_id = device_state.config.unit_id;
        let span = info_span!("discovery", unit_id);

        async move {
            let mut ctx = client.client.lock().await;
            ctx.set_slave(Slave(unit_id));
            drop(ctx);

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

            // Find supported inverter model
            let available_models = device.models.supported_model_ids();
            info!(
                "Device supports SunSpec models: {}",
                available_models
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            device_state.supported_model = Some(self.identify_inverter_model(
                unit_id,
                &available_models,
                device_state.config.preferred_model,
            )?);

            // Check if controls are enabled and identify which model to use
            device_state.active_control = self.identify_control_model(
                unit_id,
                &available_models,
                device_state.config.enable_controls,
                &device,
            );

            // Try to read Model 1 for metadata/serial
            self.read_metadata(&device, device_state).await?;

            // Read and Publish Nameplate Data
            self.read_nameplate(&device, device_state, &available_models)
                .await?;

            device_state.device = Some(device);
            Ok(())
        }
        .instrument(span)
        .await
    }

    fn identify_inverter_model(
        &self,
        unit_id: u8,
        available_models: &[u16],
        preferred: Option<u16>,
    ) -> Result<u16> {
        let mut selected_model = None;

        // Check if user has a preference
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

        // Fallback to default priority list if no preference or preference was unavailable
        if selected_model.is_none() {
            selected_model = SUPPORTED_MODELS
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
        &self,
        unit_id: u8,
        available_models: &[u16],
        enable_controls: bool,
        device: &sunspec::client::AsyncDevice<Arc<Mutex<ModbusContext>>>,
    ) -> ActiveControlModel {
        if !enable_controls {
            return ActiveControlModel::None;
        }

        if available_models.contains(&704) {
            info!("Using SunSpec Model 704 for controls on unit {}", unit_id);
            ActiveControlModel::Model704 {
                base_addr: device.models.m704.addr,
            }
        } else if available_models.contains(&123) {
            info!("Using SunSpec Model 123 for controls on unit {}", unit_id);
            ActiveControlModel::Model123 {
                base_addr: device.models.m123.addr,
            }
        } else {
            warn!(
                "Device {} has controls enabled in config, but neither Model 704 nor Model 123 is supported by hardware.",
                unit_id
            );
            ActiveControlModel::None
        }
    }

    async fn read_metadata(
        &self,
        device: &sunspec::client::AsyncDevice<Arc<Mutex<ModbusContext>>>,
        device_state: &mut DeviceState,
    ) -> Result<()> {
        let unit_id = device_state.config.unit_id;
        let m1_res = tokio::time::timeout(
            Duration::from_secs(POLL_TIMEOUT_SECS),
            device.read_model::<Model1>(),
        )
        .instrument(info_span!("model1_read", unit_id))
        .await;

        match m1_res {
            Ok(Ok(m1)) => {
                let serial = m1.sn.trim().to_string();
                let manufacturer = m1.mn.trim().to_string();
                let model = m1.md.trim().to_string();
                let version_opt = m1
                    .vr
                    .as_ref()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty());

                if serial.is_empty() {
                    warn!(
                        "Device {} returned an empty serial number, skipping discovery",
                        unit_id
                    );
                    return Ok(());
                }

                device_state.inverter_topic = Some(self.ha.inverter_topic(&serial));
                device_state.status_topic = Some(self.ha.status_topic(&serial));

                if device_state.serial.as_ref() != Some(&serial) {
                    info!(
                        "Discovered device: {} {} (Serial: {})",
                        manufacturer, model, serial
                    );
                    device_state.serial = Some(serial.clone());
                    device_state.manufacturer = Some(manufacturer.clone());
                    device_state.model = Some(model.clone());
                    device_state.version = version_opt.clone();

                    if self.ha_enabled {
                        let messages = self.ha.generate_discovery_messages(
                            &serial,
                            &manufacturer,
                            &model,
                            version_opt.as_deref(),
                            device_state.supported_model,
                            device_state.active_control,
                        );

                        for msg in messages {
                            self.mqtt_tx.send(msg).await?;
                        }
                    }
                } else {
                    info!("Refreshed connection to device (Serial: {})", serial);
                }
                Ok(())
            }
            Ok(Err(e)) => {
                warn!("Failed to read Model 1 for unit {}: {}", unit_id, e);
                Err(Pv2MqttError::DeviceDiscovery(unit_id, ModbusError::from(e)))
            }
            Err(_) => {
                warn!("Timeout reading Model 1 for unit {}", unit_id);
                Err(Pv2MqttError::Modbus(ModbusError::Timeout(format!(
                    "Timeout reading Model 1 for device {}",
                    unit_id
                ))))
            }
        }
    }

    async fn read_nameplate(
        &self,
        device: &sunspec::client::AsyncDevice<Arc<Mutex<ModbusContext>>>,
        device_state: &DeviceState,
        available_models: &[u16],
    ) -> Result<()> {
        let serial = match &device_state.serial {
            Some(s) => s,
            None => return Ok(()),
        };
        let unit_id = device_state.config.unit_id;

        let mut nameplate: Option<crate::models::NameplateData> = None;

        if available_models.contains(&702) {
            match tokio::time::timeout(
                Duration::from_secs(POLL_TIMEOUT_SECS),
                device.read_model::<sunspec::models::model702::Model702>(),
            )
            .await
            {
                Ok(Ok(m702)) => {
                    debug!("Read Model 702 nameplate for unit {}", unit_id);
                    nameplate = Some(crate::models::NameplateData {
                        w_max: m702.w_max_rtg.map(|v| v as f32),
                        va_max: m702.va_max_rtg.map(|v| v as f32),
                        var_max_inj: m702.var_max_inj_rtg.map(|v| v as f32),
                        var_max_abs: m702.var_max_abs_rtg.map(|v| v as f32),
                    })
                }
                Ok(Err(e)) => warn!("Failed to read Model 702 for unit {}: {}", unit_id, e),
                Err(_) => warn!("Timeout reading Model 702 for unit {}", unit_id),
            }
        } else if available_models.contains(&120) {
            match tokio::time::timeout(
                Duration::from_secs(POLL_TIMEOUT_SECS),
                device.read_model::<sunspec::models::model120::Model120>(),
            )
            .await
            {
                Ok(Ok(m120)) => {
                    debug!("Read Model 120 nameplate for unit {}", unit_id);
                    use crate::models::apply_sf;
                    nameplate = Some(crate::models::NameplateData {
                        w_max: apply_sf(m120.w_rtg, m120.w_rtg_sf),
                        va_max: apply_sf(m120.va_rtg, m120.va_rtg_sf),
                        var_max_inj: apply_sf(m120.v_ar_rtg_q1 as u16, m120.v_ar_rtg_sf), // Simplified mapping
                        var_max_abs: apply_sf(m120.v_ar_rtg_q4 as u16, m120.v_ar_rtg_sf),
                    });
                }
                Ok(Err(e)) => warn!("Failed to read Model 120 for unit {}: {}", unit_id, e),
                Err(_) => warn!("Timeout reading Model 120 for unit {}", unit_id),
            }
        }

        if let Some(nameplate) = nameplate {
            let prefix = device_state
                .inverter_topic
                .as_ref()
                .and_then(|t| t.split('/').next())
                .unwrap_or("solar");
            let nameplate_topic = format!("{}/inverter/{}/nameplate", prefix, serial);

            match serde_json::to_vec(&nameplate) {
                Ok(payload) => {
                    let _ = self
                        .mqtt_tx
                        .send(crate::mqtt::MqttMessage::Publish {
                            topic: nameplate_topic,
                            payload,
                            retain: true,
                        })
                        .await;
                }
                Err(e) => warn!("Failed to serialize nameplate data: {}", e),
            }

            if self.ha_enabled {
                let ha_msgs = self.ha.generate_nameplate_discovery_messages(
                    serial,
                    device_state.manufacturer.as_deref().unwrap_or(""),
                    device_state.model.as_deref().unwrap_or(""),
                    device_state.version.as_deref(),
                );
                for msg in ha_msgs {
                    let _ = self.mqtt_tx.send(msg).await;
                }
            }
        }
        Ok(())
    }
}
