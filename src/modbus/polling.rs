use super::{ConnectionTask, DeviceState, POLL_TIMEOUT_SECS};
use crate::error::{Pv2MqttError, Result};
use crate::models::{InverterData, poll_and_apply};
use crate::mqtt::MqttMessage;
use chrono::Utc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sunspec::client::AsyncDevice;
use tokio::sync::Mutex;
use tokio_modbus::client::Context as ModbusContext;
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
                        match poll_and_apply(123, device, &mut data).await {
                            Ok(_) => {}
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
                    let pv_err =
                        Pv2MqttError::Internal(format!("Timeout polling device {}", serial));
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
        let mut data = InverterData {
            timestamp: Utc::now(),
            ..Default::default()
        };
        poll_and_apply(model_id, device, &mut data).await?;
        Ok(data)
    }

    pub async fn ping_device(&self, device: &AsyncDevice<Arc<Mutex<ModbusContext>>>) -> Result<()> {
        use crate::modbus::READ_TIMEOUT_SECS;
        use tokio_modbus::Slave;
        use tokio_modbus::client::Reader;

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
            Pv2MqttError::Internal(format!("Keep-alive timeout for unit {}", device.slave_id))
        })??;

        let _ = regs_res.map_err(|e| Pv2MqttError::Internal(format!("Keep-alive error: {}", e)))?;

        Ok(())
    }
}
