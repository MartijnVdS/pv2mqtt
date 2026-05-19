// SPDX-License-Identifier: Apache-2.0

use super::{DeviceState, InverterConnection};
use crate::error::{Pv2MqttError, Result};
use crate::homeassistant::HomeAssistantIntegration;
use crate::models::ActiveControlModel;
use crate::mqtt::MqttMessage;
use bytes::BufMut;
use chrono::Utc;
use std::time::Instant;
use tracing::{error, info, warn};

// Define a type alias for the MQTT sender to keep signatures clean
type MqttTx = tokio::sync::mpsc::Sender<MqttMessage>;

impl<C: InverterConnection> super::ConnectionTask<C> {
    pub async fn perform_device_poll(
        mqtt_tx: &MqttTx,
        ha: &mut HomeAssistantIntegration,
        device_state: &mut DeviceState<C>,
        now: Instant,
    ) -> Result<()> {
        // ONLY poll if serial number and supported_model are known
        let (serial, model_id, device) = match (
            &device_state.serial,
            device_state.supported_model,
            &device_state.device,
        ) {
            (Some(s), Some(m), Some(d)) => (s, m, d),
            _ => {
                device_state.last_poll = Some(now);
                return Ok(());
            }
        };

        let mut data = crate::models::InverterData {
            timestamp: Utc::now(),
            ..Default::default()
        };

        if let Err(e) = device.poll(model_id, &mut data).await {
            let pv_err = match e {
                Pv2MqttError::Modbus(me) => Pv2MqttError::ModelRead(model_id, me),
                _ => e,
            };
            error!("Failed to poll: {}", pv_err);
            let _ = Self::report_status(mqtt_tx, ha, device_state, serial, "ERROR", Some(&pv_err))
                .await;
            return Err(pv_err);
        }

        info!("Successfully polled (Serial: {})", serial);

        // Optionally poll the active control model for status
        match device_state.active_control {
            ActiveControlModel::Model123 { .. } => {
                if let Err(e) = device.poll(123, &mut data).await {
                    warn!("Failed to poll controls (Model 123) for {}: {}", serial, e);
                }
            }
            ActiveControlModel::Model704 { .. } => {
                if let Err(e) = device.poll(704, &mut data).await {
                    warn!("Failed to poll controls (Model 704) for {}: {}", serial, e);
                }
            }
            ActiveControlModel::None => {}
        }

        device_state.last_success_timestamp = Some(Utc::now());

        device_state.serialization_buffer.clear();
        serde_json::to_writer((&mut device_state.serialization_buffer).writer(), &data)?;
        let payload = device_state.serialization_buffer.split().freeze();

        let topic = device_state
            .inverter_topic
            .clone()
            .unwrap_or_else(|| ha.inverter_topic(serial));
        mqtt_tx
            .send(MqttMessage::Publish {
                topic,
                payload,
                retain: false,
            })
            .await?;

        Self::report_status(mqtt_tx, ha, device_state, serial, "OK", None).await?;

        device_state.last_poll = Some(now);
        Ok(())
    }

    pub async fn report_status(
        mqtt_tx: &MqttTx,
        ha: &mut HomeAssistantIntegration,
        device_state: &DeviceState<C>,
        serial: &str,
        status: &str,
        error: Option<&Pv2MqttError>,
    ) -> Result<()> {
        let status_topic = device_state
            .status_topic
            .clone()
            .unwrap_or_else(|| ha.status_topic(serial));
        let status_msg = ha.generate_status_message(
            status_topic,
            status,
            error,
            device_state.last_success_timestamp.as_ref(),
        );
        mqtt_tx.send(status_msg).await.map_err(Into::into)
    }
}
