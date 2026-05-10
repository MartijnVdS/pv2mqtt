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
    pub entity_category: Option<&'static str>,
    pub state_topic: Option<String>,
}

type SensorDefinitionTuple = (
    &'static str,
    Option<&'static str>,
    Option<&'static str>,
    Option<&'static str>,
    &'static str,
);

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

        // 1. New Modern Discovery Message
        let topic = format!("{}/device/{}/config", self.ha_prefix, serial);
        let mut components = serde_json::Map::new();

        // Standard Sensors
        for (name, unit, device_class, state_class, label) in
            self.collect_sensor_definitions(model_id)
        {
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

            let entity_category = if name.starts_with("Tmp") {
                Some("diagnostic")
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
                component: Some("sensor"),
                command_topic: None,
                entity_category,
                state_topic: None,
            };
            components.insert(
                name.to_string(),
                self.modern_component_payload(serial, &ctx),
            );
        }

        // Binary Sensors
        if model_id == Some(701) {
            let binary_sensors = vec![("ConnSt", Some("connectivity"), "Grid Connection Status")];
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
                    entity_category: None,
                    state_topic: None,
                };
                components.insert(
                    name.to_string(),
                    self.modern_component_payload(serial, &ctx),
                );
            }
        }

        // Controls
        let mut controls = Vec::new();
        if active_control != crate::models::ActiveControlModel::None {
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
        }

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
                entity_category: None,
                state_topic: None,
            };
            components.insert(
                name.to_string(),
                self.modern_component_payload(serial, &ctx),
            );
        }

        // Nameplate Sensors
        let nameplate_topic = format!("{}/inverter/{}/nameplate", self.topic_prefix, serial);
        let nameplate_sensors = vec![
            (
                "WMax",
                Some("W"),
                Some("power"),
                Some("measurement"),
                "Max Active Power",
            ),
            (
                "VAMax",
                Some("VA"),
                Some("apparent_power"),
                Some("measurement"),
                "Max Apparent Power",
            ),
            (
                "VArMaxInj",
                Some("var"),
                Some("reactive_power"),
                Some("measurement"),
                "Max Reactive Power Injected",
            ),
            (
                "VArMaxAbs",
                Some("var"),
                Some("reactive_power"),
                Some("measurement"),
                "Max Reactive Power Absorbed",
            ),
        ];

        for (name, unit, device_class, state_class, label) in nameplate_sensors {
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
                enabled_by_default: true,
                options: None,
                component: Some("sensor"),
                command_topic: None,
                entity_category: Some("diagnostic"),
                state_topic: Some(nameplate_topic.clone()),
            };
            components.insert(
                name.to_string(),
                self.modern_component_payload(serial, &ctx),
            );
        }

        let payload = json!({
            "dev": {
                "ids": [serial],
                "name": format!("Inverter {}", serial),
                "mf": manufacturer,
                "mdl": model,
                "sw": version,
            },
            "o": {
                "name": "pv2mqtt",
                "sw": env!("CARGO_PKG_VERSION"),
            },
            "cmps": components,
            "state_topic": self.inverter_topic(serial),
            "qos": 1,
        });

        messages.push(MqttMessage::Publish {
            topic,
            payload: serde_json::to_vec(&payload).unwrap_or_default(),
            retain: true,
        });

        messages
    }

    fn modern_component_payload(&self, serial: &str, ctx: &DiscoveryContext) -> serde_json::Value {
        let mut payload = json!({
            "p": ctx.component.unwrap_or("sensor"),
            "name": ctx.label,
            "unique_id": format!("{}_{}_{}", self.topic_prefix, serial, ctx.name),
            "value_template": format!(
                "{{{{ value_json.{} }}}}",
                ctx.value_path.as_deref().unwrap_or(ctx.name)
            ),
            "enabled_by_default": ctx.enabled_by_default,
        });

        if let Some(state_topic) = &ctx.state_topic {
            payload["state_topic"] = json!(state_topic);
        }

        if let Some(cmd_topic) = &ctx.command_topic {
            payload["command_topic"] = json!(cmd_topic);
        }

        if ctx.component == Some("switch") {
            payload["payload_on"] = json!(true);
            payload["payload_off"] = json!(false);
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
        if let Some(ec) = ctx.entity_category {
            payload["entity_category"] = json!(ec);
        }

        payload
    }

    fn collect_sensor_definitions(&self, model_id: Option<u16>) -> Vec<SensorDefinitionTuple> {
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
            ("PF", None, Some("power_factor"), Some("measurement"), "Power Factor"),
            ("St", None, Some("enum"), None, "Status"),
        ];

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
                sensors.extend(vec![
                    (
                        "TmpAmb",
                        Some("°C"),
                        Some("temperature"),
                        Some("measurement"),
                        "Ambient Temperature",
                    ),
                    (
                        "TmpSw",
                        Some("°C"),
                        Some("temperature"),
                        Some("measurement"),
                        "IGBT/MOSFET Temperature",
                    ),
                    ("Alrm", None, None, None, "Alarms"),
                    ("MnAlrmInfo", None, None, None, "Manufacturer Alarm Info"),
                    ("DERMode", None, None, None, "DER Operational Mode"),
                    (
                        "ThrotPct",
                        Some("%"),
                        None,
                        Some("measurement"),
                        "Throttling Percentage",
                    ),
                    ("ThrotSrc", None, None, None, "Throttling Source"),
                ]);
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
        sensors
    }

    pub fn generate_nameplate_discovery_messages(
        &self,
        _serial: &str,
        _manufacturer: &str,
        _model: &str,
        _version: Option<&str>,
    ) -> Vec<MqttMessage> {
        // This is now redundant but kept for API compatibility during migration if needed.
        Vec::new()
    }

    pub fn discovery_message(&self, serial: &str, ctx: &DiscoveryContext) -> (String, Vec<u8>) {
        // Redundant with modern_component_payload, but kept for tests if they use it directly.
        let component = ctx.component.unwrap_or("sensor");
        let topic = format!(
            "{}/{}/{}/{}/config",
            self.ha_prefix, component, serial, ctx.name
        );
        let state_topic = ctx
            .state_topic
            .clone()
            .unwrap_or_else(|| self.inverter_topic(serial));
        let mut payload = json!({
            "name": ctx.label,
            "state_topic": state_topic,
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

        if let Some(ec) = ctx.entity_category {
            payload["entity_category"] = json!(ec);
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
        // - 1 Modern discovery message
        assert_eq!(msgs.len(), 1);

        // Verify the modern message
        let modern_msg = &msgs[0];
        let MqttMessage::Publish { topic, payload, .. } = modern_msg;
        assert_eq!(topic, "homeassistant/device/SN123/config");
        let body: serde_json::Value = serde_json::from_slice(payload).unwrap();
        assert_eq!(body["dev"]["ids"][0], "SN123");
        assert!(body["cmps"].as_object().unwrap().contains_key("W"));
        assert!(body["cmps"].as_object().unwrap().contains_key("WMax"));
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
        // - 1 Modern
        assert_eq!(msgs.len(), 1);

        let modern_msg = &msgs[0];
        let MqttMessage::Publish { payload, .. } = modern_msg;
        let body: serde_json::Value = serde_json::from_slice(payload).unwrap();
        let cmps = body["cmps"].as_object().unwrap();
        assert!(cmps.contains_key("PhVphC"));
        assert!(cmps.contains_key("Conn"));
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
        // - 1 Modern
        assert_eq!(msgs.len(), 1);

        let modern_msg = &msgs[0];
        let MqttMessage::Publish { payload, .. } = modern_msg;
        let body: serde_json::Value = serde_json::from_slice(payload).unwrap();
        let cmps = body["cmps"].as_object().unwrap();
        assert!(cmps.contains_key("WMaxLimPct"));
        assert!(!cmps.contains_key("Conn")); // 704 doesn't have Conn
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
        // - 1 Modern
        assert_eq!(msgs.len(), 1);

        let modern_msg = &msgs[0];
        let MqttMessage::Publish { payload, .. } = modern_msg;
        let body: serde_json::Value = serde_json::from_slice(payload).unwrap();
        let cmps = body["cmps"].as_object().unwrap();
        assert!(cmps.contains_key("ConnSt"));
        assert!(cmps.contains_key("Alrm"));
    }

    #[test]
    fn test_generate_nameplate_discovery_messages() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let msgs =
            ha.generate_nameplate_discovery_messages("SN123", "Brand", "ModelX", Some("v1.0"));

        // Now empty as it's merged into main discovery
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn test_modern_payload_structure_details() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let msgs = ha.generate_discovery_messages(
            "SN123",
            "Brand",
            "ModelX",
            Some("v1.0"),
            Some(101),
            crate::models::ActiveControlModel::None,
        );

        let modern_msg = &msgs[0];
        let MqttMessage::Publish { payload, .. } = modern_msg;
        let body: serde_json::Value = serde_json::from_slice(payload).unwrap();

        // Check state topics
        assert_eq!(body["state_topic"], "solar/inverter/SN123");

        let cmps = body["cmps"].as_object().unwrap();

        // W should inherit state_topic
        assert!(cmps["W"]["state_topic"].is_null());

        // WMax (nameplate) should override state_topic
        assert_eq!(
            cmps["WMax"]["state_topic"],
            "solar/inverter/SN123/nameplate"
        );
        assert_eq!(cmps["WMax"]["entity_category"], "diagnostic");

        // Temperature sensors should be diagnostic
        assert_eq!(cmps["TmpCab"]["entity_category"], "diagnostic");

        // Check unique_id format
        assert_eq!(cmps["W"]["unique_id"], "solar_SN123_W");
    }
}
