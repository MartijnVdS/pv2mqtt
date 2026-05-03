use super::{ConnectionTask, DiscoveryContext, MqttMessage};
use crate::error::Pv2MqttError;
use crate::error::Result;
use chrono::{DateTime, Utc};

impl ConnectionTask {
    pub fn inverter_topic(&self, serial: &str) -> String {
        format!("{}/inverter/{}", self.topic_prefix, serial)
    }

    pub fn status_message(
        &self,
        serial: &str,
        status: &str,
        error: Option<&Pv2MqttError>,
        last_success: Option<&DateTime<Utc>>,
    ) -> (String, String) {
        let topic = format!("{}/inverter/{}/status", self.topic_prefix, serial);
        let payload = serde_json::json!({
            "timestamp": last_success.map(|dt| dt.to_rfc3339()),
            "status": status,
            "error": error.as_ref().map(|e| e.to_string()),
            "error_category": error.as_ref().map(|e| e.category()),
        })
        .to_string();
        (topic, payload)
    }

    pub async fn publish_discovery(
        &self,
        serial: &str,
        manufacturer: &str,
        model: &str,
        version: Option<&str>,
        model_id: Option<u16>,
        enable_controls: bool,
    ) -> Result<()> {
        let mut sensors = vec![
            ("W", Some("W"), Some("power"), Some("measurement"), "Power"),
            (
                "WH",
                Some("Wh"),
                Some("energy"),
                Some("total_increasing"),
                "Energy",
            ),
            (
                "Hz",
                Some("Hz"),
                Some("frequency"),
                Some("measurement"),
                "Frequency",
            ),
            (
                "TmpCab",
                Some("°C"),
                Some("temperature"),
                Some("measurement"),
                "Cabinet Temperature",
            ),
            (
                "TmpSnk",
                Some("°C"),
                Some("temperature"),
                Some("measurement"),
                "Heat Sink Temperature",
            ),
            ("St", None, Some("enum"), None, "Status"),
        ];

        // Add phase-specific sensors based on model
        if let Some(id) = model_id {
            // All supported models have at least Phase A
            sensors.push((
                "PhVphA",
                Some("V"),
                Some("voltage"),
                Some("measurement"),
                "Voltage Phase A",
            ));
            sensors.push((
                "AphA",
                Some("A"),
                Some("current"),
                Some("measurement"),
                "Current Phase A",
            ));

            if matches!(id, 102 | 103 | 112 | 113) {
                sensors.push((
                    "PhVphB",
                    Some("V"),
                    Some("voltage"),
                    Some("measurement"),
                    "Voltage Phase B",
                ));
                sensors.push((
                    "AphB",
                    Some("A"),
                    Some("current"),
                    Some("measurement"),
                    "Current Phase B",
                ));
            }

            if matches!(id, 103 | 113) {
                sensors.push((
                    "PhVphC",
                    Some("V"),
                    Some("voltage"),
                    Some("measurement"),
                    "Voltage Phase C",
                ));
                sensors.push((
                    "AphC",
                    Some("A"),
                    Some("current"),
                    Some("measurement"),
                    "Current Phase C",
                ));
            }
        }

        for (name, unit, device_class, state_class, label) in sensors {
            let enabled_by_default = matches!(name, "W" | "WH" | "St");
            let options = if name == "St" {
                Some(vec![
                    "OFF",
                    "SLEEPING",
                    "STARTING",
                    "MPPT",
                    "THROTTLED",
                    "SHUTTING_DOWN",
                    "FAULT",
                    "STANDBY",
                    "UNKNOWN",
                ])
            } else {
                None
            };

            let ctx = DiscoveryContext {
                manufacturer,
                model,
                version,
                name,
                value_path: None,
                unit,
                device_class,
                state_class,
                label,
                enabled_by_default,
                options,
                component: None,
                command_topic: None,
            };
            let (topic, payload) = self.discovery_message(serial, &ctx);
            self.mqtt_tx
                .send(MqttMessage::Publish {
                    topic,
                    payload,
                    retain: true,
                })
                .await?;
        }

        // Add control entities if Model 123 is enabled
        if enable_controls {
            let controls = vec![
                (
                    "Conn",
                    "switch",
                    None,
                    None,
                    "Connection",
                    format!("{}/inverter/{}/set/Conn", self.topic_prefix, serial),
                ),
                (
                    "WMaxLimPct",
                    "number",
                    Some("power_factor"),
                    None,
                    "Active Power Limit",
                    format!("{}/inverter/{}/set/WMaxLimPct", self.topic_prefix, serial),
                ),
                (
                    "WMaxLim_Ena",
                    "switch",
                    None,
                    None,
                    "Active Power Limit Enable",
                    format!("{}/inverter/{}/set/WMaxLim_Ena", self.topic_prefix, serial),
                ),
            ];

            for (name, component, device_class, state_class, label, cmd_topic) in controls {
                let ctx = DiscoveryContext {
                    manufacturer,
                    model,
                    version,
                    name,
                    value_path: Some(format!("Controls.{}", name)),
                    unit: if name == "WMaxLimPct" {
                        Some("%")
                    } else {
                        None
                    },
                    device_class,
                    state_class,
                    label,
                    enabled_by_default: true,
                    options: None,
                    component: Some(component),
                    command_topic: Some(cmd_topic),
                };
                let (topic, payload) = self.discovery_message(serial, &ctx);
                self.mqtt_tx
                    .send(MqttMessage::Publish {
                        topic,
                        payload,
                        retain: true,
                    })
                    .await?;
            }
        }

        Ok(())
    }

    pub fn discovery_message(&self, serial: &str, ctx: &DiscoveryContext) -> (String, String) {
        let component = ctx.component.unwrap_or("sensor");
        let topic = format!(
            "{}/{}/{}/{}/config",
            self.ha_prefix, component, serial, ctx.name
        );
        let mut payload = serde_json::json!({
            "name": ctx.label,
            "state_topic": self.inverter_topic(serial),
            "value_template": format!(
                "{{{{ value_json.{} }}}}",
                ctx.value_path.as_deref().unwrap_or(ctx.name)
            ),
            "unique_id": format!("{}_{}_{}", self.topic_prefix, serial, ctx.name),
            "force_update": true,
            "enabled_by_default": ctx.enabled_by_default,
            "device": {
                "identifiers": [serial],
                "name": format!("Inverter {}", serial),
                "manufacturer": ctx.manufacturer,
                "model": ctx.model,
            }
        });

        if let Some(cmd_topic) = &ctx.command_topic {
            payload["command_topic"] = serde_json::json!(cmd_topic);
        }

        if component == "switch" {
            payload["payload_on"] = serde_json::json!(true);
            payload["payload_off"] = serde_json::json!(false);
        }

        if let Some(v) = ctx.version {
            payload["device"]["sw_version"] = serde_json::json!(v);
        }

        if let Some(unit) = ctx.unit {
            payload["unit_of_measurement"] = serde_json::json!(unit);
        }
        if let Some(dc) = ctx.device_class {
            payload["device_class"] = serde_json::json!(dc);
        }
        if let Some(sc) = ctx.state_class {
            payload["state_class"] = serde_json::json!(sc);
        }
        if let Some(options) = &ctx.options {
            payload["options"] = serde_json::json!(options);
        }

        (topic, payload.to_string())
    }
}
