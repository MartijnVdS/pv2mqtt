use super::{ConnectionTask, DeviceState, POLL_TIMEOUT_SECS, READ_TIMEOUT_SECS};
use crate::error::{ModbusError, Pv2MqttError, Result};
use crate::models::InverterData;
use crate::mqtt::MqttMessage;
use chrono::Utc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sunspec::client::AsyncDevice;
use tokio::sync::Mutex;
use tokio_modbus::Slave;
use tokio_modbus::client::{Context as ModbusContext, Reader};
use tokio_modbus::slave::SlaveContext;
use tracing::{Instrument, debug, error, info, info_span, warn};

impl ConnectionTask {
    pub async fn perform_device_poll(
        &self,
        device_state: &mut DeviceState,
        now: Instant,
    ) -> Result<()> {
        let span = info_span!("poll", unit_id = device_state.config.unit_id);
        self.perform_device_poll_internal(device_state, now)
            .instrument(span)
            .await
    }

    async fn perform_device_poll_internal(
        &self,
        device_state: &mut DeviceState,
        now: Instant,
    ) -> Result<()> {
        // ONLY poll if serial number and supported_model are known
        if let (Some(serial), Some(model_id), Some(device)) = (
            &device_state.serial,
            device_state.supported_model,
            &device_state.device,
        ) {
            let poll_result = tokio::time::timeout(
                Duration::from_secs(POLL_TIMEOUT_SECS),
                self.poll_device(device, model_id),
            )
            .await;

            match poll_result {
                Ok(Ok(mut data)) => {
                    info!("Successfully polled (Serial: {})", serial);

                    // Optionally poll Model 123 for controls status
                    if device_state.config.enable_controls {
                        match self.poll_model123(device).await {
                            Ok(c) => data.controls = Some(c),
                            Err(e) => {
                                warn!("Failed to poll controls (Model 123) for {}: {}", serial, e);
                            }
                        }
                    }

                    device_state.last_success_timestamp = Some(Utc::now());
                    let payload = serde_json::to_string(&data)?;
                    let topic = self.inverter_topic(serial);
                    self.mqtt_tx
                        .send(MqttMessage::Publish {
                            topic,
                            payload,
                            retain: false,
                        })
                        .await?;

                    let (status_topic, status_payload) = self.status_message(
                        serial,
                        "OK",
                        None,
                        device_state.last_success_timestamp.as_ref(),
                    );
                    self.mqtt_tx
                        .send(MqttMessage::Publish {
                            topic: status_topic,
                            payload: status_payload,
                            retain: true,
                        })
                        .await?;
                }
                Ok(Err(e)) => {
                    let pv_err = match e {
                        Pv2MqttError::Modbus(me) => Pv2MqttError::ModelRead(model_id, me),
                        _ => e,
                    };
                    error!("Failed to poll: {}", pv_err);
                    // Update status with error
                    let (status_topic, status_payload) = self.status_message(
                        serial,
                        "ERROR",
                        Some(&pv_err),
                        device_state.last_success_timestamp.as_ref(),
                    );
                    let _ = self
                        .mqtt_tx
                        .send(MqttMessage::Publish {
                            topic: status_topic,
                            payload: status_payload,
                            retain: true,
                        })
                        .await;
                    return Err(pv_err);
                }
                Err(_) => {
                    let pv_err = Pv2MqttError::Modbus(ModbusError::Timeout(format!(
                        "Timeout polling device {}",
                        serial
                    )));
                    error!("{}", pv_err);
                    let (status_topic, status_payload) = self.status_message(
                        serial,
                        "ERROR",
                        Some(&pv_err),
                        device_state.last_success_timestamp.as_ref(),
                    );
                    let _ = self
                        .mqtt_tx
                        .send(MqttMessage::Publish {
                            topic: status_topic,
                            payload: status_payload,
                            retain: true,
                        })
                        .await;
                    return Err(pv_err);
                }
            }
        }

        device_state.last_poll = Some(now);
        Ok(())
    }

    pub async fn poll_device(
        &self,
        device: &AsyncDevice<Arc<Mutex<ModbusContext>>>,
        model_id: u16,
    ) -> Result<InverterData> {
        use sunspec::models::{
            model101::Model101, model102::Model102, model103::Model103, model111::Model111,
            model112::Model112, model113::Model113,
        };
        match model_id {
            101 => Ok(device
                .read_model::<Model101>()
                .await
                .map_err(Pv2MqttError::from)?
                .into()),
            102 => Ok(device
                .read_model::<Model102>()
                .await
                .map_err(Pv2MqttError::from)?
                .into()),
            103 => Ok(device
                .read_model::<Model103>()
                .await
                .map_err(Pv2MqttError::from)?
                .into()),
            111 => Ok(device
                .read_model::<Model111>()
                .await
                .map_err(Pv2MqttError::from)?
                .into()),
            112 => Ok(device
                .read_model::<Model112>()
                .await
                .map_err(Pv2MqttError::from)?
                .into()),
            113 => Ok(device
                .read_model::<Model113>()
                .await
                .map_err(Pv2MqttError::from)?
                .into()),
            _ => Err(Pv2MqttError::UnsupportedModel(model_id)),
        }
    }

    pub async fn poll_model123(
        &self,
        device: &AsyncDevice<Arc<Mutex<ModbusContext>>>,
    ) -> Result<crate::models::Model123Data> {
        let m123 = device
            .read_model::<sunspec::models::model123::Model123>()
            .await
            .map_err(Pv2MqttError::from)?;
        Ok(m123.into())
    }

    pub async fn ping_device(&self, device: &AsyncDevice<Arc<Mutex<ModbusContext>>>) -> Result<()> {
        debug!("Sending keep-alive");
        let addr = device.models.m1.addr;
        let mut ctx = device.client.lock().await;
        ctx.set_slave(Slave(device.slave_id));
        let regs_res = tokio::time::timeout(
            Duration::from_secs(READ_TIMEOUT_SECS),
            ctx.read_holding_registers(addr, 2),
        )
        .await
        .map_err(|_| {
            ModbusError::Timeout(format!("Keep-alive timeout for unit {}", device.slave_id))
        })??;

        let _ =
            regs_res.map_err(|e| ModbusError::Protocol(format!("Keep-alive exception: {}", e)))?;

        Ok(())
    }
}
