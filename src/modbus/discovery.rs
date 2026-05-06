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
            info!("Device supports SunSpec models: {}", available_models
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", "));

            let mut selected_model = None;

            // Check if user has a preference
            if let Some(preferred) = device_state.config.preferred_model {
                if available_models.contains(&preferred) {
                    info!("Using preferred SunSpec model {} for unit {}", preferred, unit_id);
                    selected_model = Some(preferred);
                } else {
                    warn!(
                        "Preferred SunSpec model {} is not supported by hardware for unit {}. Falling back to default priority list.",
                        preferred, unit_id
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

            device_state.supported_model = selected_model;

            if device_state.supported_model.is_none() {
                let available = available_models
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(Pv2MqttError::Internal(format!(
                    "No supported inverter model found. Available: {}",
                    available
                )));
            }

            // Check if controls are enabled and identify which model to use
            device_state.active_control = ActiveControlModel::None;
            if device_state.config.enable_controls {
                if available_models.contains(&704) {
                    info!("Using SunSpec Model 704 for controls on unit {}", unit_id);
                    device_state.active_control = ActiveControlModel::Model704 {
                        base_addr: device.models.m704.addr,
                    };
                } else if available_models.contains(&123) {
                    info!("Using SunSpec Model 123 for controls on unit {}", unit_id);
                    device_state.active_control = ActiveControlModel::Model123 {
                        base_addr: device.models.m123.addr,
                    };
                } else {
                    warn!(
                        "Device {} has controls enabled in config, but neither Model 704 nor Model 123 is supported by hardware.",
                        unit_id
                    );
                }
            }

            // Try to read Model 1 for metadata/serial
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
                }
                Ok(Err(e)) => {
                    warn!("Failed to read Model 1 for unit {}: {}", unit_id, e);
                    return Err(Pv2MqttError::DeviceDiscovery(unit_id, ModbusError::from(e)));
                }
                Err(_) => {
                    warn!("Timeout reading Model 1 for unit {}", unit_id);
                    return Err(Pv2MqttError::Modbus(ModbusError::Timeout(format!(
                        "Timeout reading Model 1 for device {}",
                        unit_id
                    ))));
                }
            }

            device_state.device = Some(device);
            Ok(())
        }
        .instrument(span)
        .await
    }
}
