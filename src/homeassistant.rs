// SPDX-License-Identifier: Apache-2.0

use crate::error::Pv2MqttError;
use crate::mqtt::MqttMessage;
use chrono::{DateTime, Utc};
use serde_json::json;

pub struct HomeAssistantIntegration {
    topic_prefix: String,
    ha_prefix: String,
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

impl HomeAssistantIntegration {
    pub fn new(topic_prefix: String, ha_prefix: String) -> Self {
        Self {
            topic_prefix,
            ha_prefix,
        }
    }

    pub fn inverter_topic(&self, serial: &str) -> String {
        format!("{}/inverter/{}", self.topic_prefix, serial)
    }

    pub fn status_topic(&self, serial: &str) -> String {
        format!("{}/inverter/{}/status", self.topic_prefix, serial)
    }

    pub fn generate_status_message(
        &self,
        topic: String,
        status: &str,
        error: Option<&Pv2MqttError>,
        last_success: Option<&DateTime<Utc>>,
    ) -> MqttMessage {
        let payload = serde_json::to_vec(&json!({
            "timestamp": last_success.map(|dt| dt.to_rfc3339()),
            "status": status,
            "error": error.as_ref().map(|e| e.to_string()),
            "error_category": error.as_ref().map(|e| e.category()),
        }))
        .unwrap_or_default();

        MqttMessage::Publish {
            topic,
            payload,
            retain: true,
        }
    }

    pub fn generate_discovery_messages(
        &self,
        serial: &str,
        manufacturer: &str,
        model: &str,
        version: Option<&str>,
        model_id: Option<u16>,
        active_control: crate::models::ActiveControlModel,
    ) -> Vec<MqttMessage> {
        let mut messages = Vec::new();
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

            if id == 701 {
                sensors.push((
                    "TmpAmb",
                    Some("°C"),
                    Some("temperature"),
                    Some("measurement"),
                    "Ambient Temperature",
                ));
                sensors.push((
                    "TmpSw",
                    Some("°C"),
                    Some("temperature"),
                    Some("measurement"),
                    "IGBT/MOSFET Temperature",
                ));
                sensors.push(("Alrm", None, None, None, "Alarms"));
                sensors.push(("MnAlrmInfo", None, None, None, "Manufacturer Alarm Info"));
                sensors.push(("DERMode", None, None, None, "DER Operational Mode"));
                sensors.push((
                    "ThrotPct",
                    Some("%"),
                    None,
                    Some("measurement"),
                    "Throttling Percentage",
                ));
                sensors.push(("ThrotSrc", None, None, None, "Throttling Source"));
            }

            if matches!(id, 102 | 103 | 112 | 113 | 701) {
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

            if matches!(id, 103 | 113 | 701) {
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
            messages.push(MqttMessage::Publish {
                topic,
                payload,
                retain: true,
            });
        }

        // Add binary sensors
        let mut binary_sensors = Vec::new();
        if model_id == Some(701) {
            binary_sensors.push(("ConnSt", Some("connectivity"), "Grid Connection Status"));
        }

        for (name, device_class, label) in binary_sensors {
            let ctx = DiscoveryContext {
                manufacturer,
                model,
                version,
                name,
                value_path: None,
                unit: None,
                device_class,
                state_class: None,
                label,
                enabled_by_default: true,
                options: None,
                component: Some("binary_sensor"),
                command_topic: None,
            };
            let (topic, payload) = self.discovery_message(serial, &ctx);
            messages.push(MqttMessage::Publish {
                topic,
                payload,
                retain: true,
            });
        }

        if active_control != crate::models::ActiveControlModel::None {
            let mut controls = Vec::new();

            if matches!(
                active_control,
                crate::models::ActiveControlModel::Model123 { .. }
            ) {
                controls.push((
                    "Conn",
                    "switch",
                    None,
                    None,
                    "Connection",
                    format!("{}/inverter/{}/set/Conn", self.topic_prefix, serial),
                ));
            }

            controls.extend(vec![
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
            ]);

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
                messages.push(MqttMessage::Publish {
                    topic,
                    payload,
                    retain: true,
                });
            }
        } else {
            let cleanup_controls = vec![
                ("Conn", "switch"),
                ("WMaxLimPct", "number"),
                ("WMaxLim_Ena", "switch"),
            ];
            for (name, component) in cleanup_controls {
                let topic = format!(
                    "{}/{}/{}/{}/config",
                    self.ha_prefix, component, serial, name
                );
                messages.push(MqttMessage::Publish {
                    topic,
                    payload: Vec::new(),
                    retain: true,
                });
            }
        }

        messages
    }

    pub fn discovery_message(&self, serial: &str, ctx: &DiscoveryContext) -> (String, Vec<u8>) {
        let component = ctx.component.unwrap_or("sensor");
        let topic = format!(
            "{}/{}/{}/{}/config",
            self.ha_prefix, component, serial, ctx.name
        );
        let mut payload = json!({
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
            payload["command_topic"] = json!(cmd_topic);
        }

        if component == "switch" {
            payload["payload_on"] = json!(true);
            payload["payload_off"] = json!(false);
        }

        if let Some(v) = ctx.version {
            payload["device"]["sw_version"] = json!(v);
        }

        if let Some(unit) = ctx.unit {
            payload["unit_of_measurement"] = json!(unit);
        }
        if let Some(dc) = ctx.device_class {
            payload["device_class"] = json!(dc);
        }
        if let Some(sc) = ctx.state_class {
            payload["state_class"] = json!(sc);
        }
        if let Some(options) = &ctx.options {
            payload["options"] = json!(options);
        }

        (topic, serde_json::to_vec(&payload).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_discovery_messages_m101() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let msgs = ha.generate_discovery_messages(
            "SN123",
            "Manufacturer",
            "Model",
            Some("1.0"),
            Some(101),
            crate::models::ActiveControlModel::None,
        );

        // Calculation:
        // - Core sensors (W, WH, Hz, TmpCab, TmpSnk, St): 6
        // - Phase A specific (PhVphA, AphA): 2
        // - Cleanup for disabled controls (Conn, WMaxLimPct, WMaxLim_Ena): 3
        // Total: 6 + 2 + 3 = 11
        assert_eq!(msgs.len(), 11);

        // Verify no Phase B/C
        for msg in &msgs {
            let MqttMessage::Publish { topic, .. } = msg;
            assert!(!topic.contains("PhVphB"));
            assert!(!topic.contains("AphB"));
            assert!(!topic.contains("PhVphC"));
            assert!(!topic.contains("AphC"));
        }
    }

    #[test]
    fn test_generate_discovery_messages_m103_with_controls() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let msgs = ha.generate_discovery_messages(
            "SN123",
            "Manufacturer",
            "Model",
            None,
            Some(103),
            crate::models::ActiveControlModel::Model123 { base_addr: 40000 },
        );

        // Calculation:
        // - Core sensors: 6
        // - Phase A: 2
        // - Phase B: 2
        // - Phase C: 2
        // - Controls (Conn, WMaxLimPct, WMaxLim_Ena): 3
        // Total: 6 + 2 + 2 + 2 + 3 = 15
        assert_eq!(msgs.len(), 15);

        // Verify presence of Phase C and Controls
        let mut has_ph_c = false;
        let mut has_conn = false;
        for msg in &msgs {
            let MqttMessage::Publish { topic, .. } = msg;
            if topic.contains("PhVphC") {
                has_ph_c = true;
            }
            if topic.contains("Conn") {
                has_conn = true;
            }
        }
        assert!(has_ph_c);
        assert!(has_conn);
    }

    #[test]
    fn test_generate_discovery_messages_m704_with_controls() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let msgs = ha.generate_discovery_messages(
            "SN123",
            "Manufacturer",
            "Model",
            None,
            Some(103),
            crate::models::ActiveControlModel::Model704 { base_addr: 40000 },
        );

        // Calculation:
        // - Core sensors: 6
        // - Phase A: 2
        // - Phase B: 2
        // - Phase C: 2
        // - Controls (WMaxLimPct, WMaxLim_Ena): 2 (NO Conn)
        // Total: 6 + 2 + 2 + 2 + 2 = 14
        assert_eq!(msgs.len(), 14);

        // Verify NO Conn
        for msg in &msgs {
            let MqttMessage::Publish { topic, .. } = msg;
            assert!(!topic.contains("Conn"));
        }
    }

    #[test]
    fn test_generate_discovery_messages_cleanup() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let msgs = ha.generate_discovery_messages(
            "SN123",
            "Manufacturer",
            "Model",
            None,
            Some(101),
            crate::models::ActiveControlModel::None, // Controls disabled -> should generate cleanup
        );

        // Calculation:
        // - Core sensors: 6
        // - Phase A: 2
        // - Cleanup: 3
        // Total: 6 + 2 + 3 = 11
        assert_eq!(msgs.len(), 11);

        let cleanup_topics: Vec<_> = msgs
            .iter()
            .filter_map(|msg| {
                let MqttMessage::Publish { topic, payload, .. } = msg;
                if payload.is_empty() {
                    Some(topic.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(cleanup_topics.len(), 3);
        assert!(cleanup_topics.iter().any(|t| t.contains("Conn")));
    }

    #[test]
    fn test_generate_discovery_messages_m701() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let msgs = ha.generate_discovery_messages(
            "SN123",
            "Manufacturer",
            "Model",
            Some("1.0"),
            Some(701),
            crate::models::ActiveControlModel::None,
        );

        // Calculation:
        // - Core sensors: 6
        // - Phase A/B/C: 2 * 3 = 6
        // - 701 Specific sensors: 7
        // - Binary sensor: 1
        // - Cleanup: 3
        // Total: 6 + 6 + 7 + 1 + 3 = 23
        assert_eq!(msgs.len(), 23);

        let mut has_conn_st = false;
        let mut has_alrm = false;
        let mut has_ph_c = false;
        for msg in &msgs {
            let MqttMessage::Publish { topic, .. } = msg;
            if topic.contains("ConnSt") {
                has_conn_st = true;
            }
            if topic.contains("Alrm") {
                has_alrm = true;
            }
            if topic.contains("PhVphC") {
                has_ph_c = true;
            }
        }
        assert!(has_conn_st);
        assert!(has_alrm);
        assert!(has_ph_c);
    }
}
