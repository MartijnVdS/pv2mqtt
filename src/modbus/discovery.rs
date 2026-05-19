// SPDX-License-Identifier: Apache-2.0

use super::{DeviceState, InverterConnection};
use crate::error::Result;
use crate::homeassistant::HomeAssistantIntegration;
use crate::mqtt::MqttMessage;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_modbus::client::Context as ModbusContext;
use tracing::{Instrument, info, info_span};

type MqttTx = tokio::sync::mpsc::Sender<MqttMessage>;

impl<C: InverterConnection> super::ConnectionTask<C> {
    pub async fn discover_device(
        mqtt_tx: &MqttTx,
        ha: &HomeAssistantIntegration,
        ha_enabled: bool,
        ctx: Arc<Mutex<ModbusContext>>,
        device_state: &mut DeviceState<C>,
    ) -> Result<()> {
        let unit_id = device_state.config.unit_id;

        let device = C::discover(ctx, unit_id, &device_state.config).await?;

        // Try to read metadata/serial
        let metadata = device
            .read_metadata()
            .instrument(info_span!("metadata_read", unit_id))
            .await?;

        if metadata.serial.is_empty() {
            tracing::warn!(
                "Device {} returned an empty serial number, skipping discovery",
                unit_id
            );
            return Ok(());
        }

        device_state.supported_model = Some(metadata.supported_model);
        device_state.active_control = metadata.active_control;

        device_state.inverter_topic = Some(ha.inverter_topic(&metadata.serial));
        device_state.status_topic = Some(ha.status_topic(&metadata.serial));
        device_state.discovery_topic = Some(ha.discovery_topic(&metadata.serial));
        device_state.nameplate_topic = Some(ha.nameplate_topic(&metadata.serial));

        if device_state.serial.as_ref() != Some(&metadata.serial) {
            info!(
                "Discovered device: {} {} (Serial: {})",
                metadata.manufacturer, metadata.model, metadata.serial
            );
            device_state.serial = Some(metadata.serial.clone());
            device_state.manufacturer = Some(metadata.manufacturer.clone());
            device_state.model = Some(metadata.model.clone());
            device_state.version = metadata.version.clone();

            if ha_enabled {
                let messages = ha.generate_discovery_messages(
                    &metadata.serial,
                    &metadata.manufacturer,
                    &metadata.model,
                    metadata.version.as_deref(),
                    device_state.supported_model,
                    device_state.active_control,
                );

                for msg in messages {
                    mqtt_tx.send(msg).await?;
                }
            }
        } else {
            info!(
                "Refreshed connection to device (Serial: {})",
                metadata.serial
            );
        }

        // Read and Publish Nameplate Data
        let available_models = device.supported_model_ids();
        if let Some(nameplate) = device.read_nameplate(&available_models).await? {
            let nameplate_topic = device_state
                .nameplate_topic
                .clone()
                .unwrap_or_else(|| ha.nameplate_topic(&metadata.serial));

            if let Ok(payload) = serde_json::to_vec(&nameplate) {
                let _ = mqtt_tx
                    .send(crate::mqtt::MqttMessage::Publish {
                        topic: nameplate_topic,
                        payload: payload.into(),
                        retain: true,
                    })
                    .await;
            }
        }

        device_state.device = Some(Arc::new(device));
        Ok(())
    }
}
