// SPDX-License-Identifier: Apache-2.0

use crate::error::Pv2MqttError;
use crate::mqtt::MqttMessage;
use chrono::{DateTime, Utc};
use serde_json::json;

pub struct HomeAssistantIntegration {
    topic_prefix: String,
    ha_prefix: String,
}

pub struct SensorDefinition {
    pub name: &'static str,
    pub label: &'static str,
    pub unit: Option<&'static str>,
    pub device_class: Option<&'static str>,
    pub state_class: Option<&'static str>,
    pub entity_category: Option<&'static str>,
    pub enabled_by_default: bool,
}

impl SensorDefinition {
    pub fn new(name: &'static str, label: &'static str) -> Self {
        let enabled_by_default = matches!(
            name,
            "W" | "WH" | "St" | "WMax" | "VAMax" | "VArMaxInj" | "VArMaxAbs"
        );
        let entity_category = if name.starts_with("Tmp") {
            Some("diagnostic")
        } else {
            None
        };

        Self {
            name,
            label,
            unit: None,
            device_class: None,
            state_class: None,
            entity_category,
            enabled_by_default,
        }
    }

    pub fn unit(mut self, unit: &'static str) -> Self {
        self.unit = Some(unit);
        self
    }

    pub fn device_class(mut self, dc: &'static str) -> Self {
        self.device_class = Some(dc);
        self
    }

    pub fn state_class(mut self, sc: &'static str) -> Self {
        self.state_class = Some(sc);
        self
    }

    pub fn entity_category(mut self, ec: &'static str) -> Self {
        self.entity_category = Some(ec);
        self
    }

    pub fn enabled_by_default(mut self, enabled: bool) -> Self {
        self.enabled_by_default = enabled;
        self
    }
}

pub struct ControlDefinition {
    pub name: &'static str,
    pub label: &'static str,
    pub component: &'static str,
    pub device_class: Option<&'static str>,
    pub state_class: Option<&'static str>,
    pub unit: Option<&'static str>,
}

impl ControlDefinition {
    pub fn new(name: &'static str, label: &'static str, component: &'static str) -> Self {
        Self {
            name,
            label,
            component,
            device_class: None,
            state_class: None,
            unit: None,
        }
    }

    pub fn device_class(mut self, dc: &'static str) -> Self {
        self.device_class = Some(dc);
        self
    }

    pub fn state_class(mut self, sc: &'static str) -> Self {
        self.state_class = Some(sc);
        self
    }

    pub fn unit(mut self, unit: &'static str) -> Self {
        self.unit = Some(unit);
        self
    }
}

pub struct DiscoveryContext<'a> {
    pub manufacturer: &'a str,
    pub model: &'a str,
    pub version: Option<&'a str>,
    pub name: &'a str,
    pub value_path: Option<String>,
    pub unit: Option<&'static str>,
    pub device_class: Option<&'static str>,
    pub state_class: Option<&'static str>,
    pub label: &'a str,
    pub enabled_by_default: bool,
    pub options: Option<Vec<&'static str>>,
    pub component: Option<&'static str>,
    pub command_topic: Option<String>,
    pub entity_category: Option<&'static str>,
    pub state_topic: Option<String>,
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
        .map(bytes::Bytes::from)
        .unwrap_or_else(|_| bytes::Bytes::new());

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
        for sensor in self.collect_sensor_definitions(model_id) {
            let options = if sensor.name == "St" {
                Some(vec![
                    "OFF",
                    "SLEEPING",
                    "STARTING",
                    "MPPT",    // model 1x1-1x3
                    "RUNNING", // model 701
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
                name: sensor.name,
                value_path: None,
                unit: sensor.unit,
                device_class: sensor.device_class,
                state_class: sensor.state_class,
                label: sensor.label,
                enabled_by_default: sensor.enabled_by_default,
                options,
                component: Some("sensor"),
                command_topic: None,
                entity_category: sensor.entity_category,
                state_topic: None,
            };
            components.insert(
                sensor.name.to_string(),
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
                controls.push(ControlDefinition::new("Conn", "Connection", "switch"));
            }

            controls.extend(vec![
                ControlDefinition::new("WMaxLimPct", "Active Power Limit", "number").unit("%"),
                ControlDefinition::new("WMaxLim_Ena", "Active Power Limit Enable", "switch"),
            ]);
        }

        for control in controls {
            let ctx = DiscoveryContext {
                manufacturer,
                model,
                version,
                name: control.name,
                value_path: Some(format!("Controls.{}", control.name)),
                unit: control.unit,
                device_class: control.device_class,
                state_class: control.state_class,
                label: control.label,
                enabled_by_default: true,
                options: None,
                component: Some(control.component),
                command_topic: Some(format!(
                    "{}/inverter/{}/set/{}",
                    self.topic_prefix, serial, control.name
                )),
                entity_category: None,
                state_topic: None,
            };
            components.insert(
                control.name.to_string(),
                self.modern_component_payload(serial, &ctx),
            );
        }

        // Nameplate Sensors
        let nameplate_topic = format!("{}/inverter/{}/nameplate", self.topic_prefix, serial);
        let nameplate_sensors = vec![
            SensorDefinition::new("WMax", "Max Active Power")
                .unit("W")
                .device_class("power")
                .state_class("measurement")
                .entity_category("diagnostic"),
            SensorDefinition::new("VAMax", "Max Apparent Power")
                .unit("VA")
                .device_class("apparent_power")
                .state_class("measurement")
                .entity_category("diagnostic"),
            SensorDefinition::new("VArMaxInj", "Max Reactive Power Injected")
                .unit("var")
                .device_class("reactive_power")
                .state_class("measurement")
                .entity_category("diagnostic"),
            SensorDefinition::new("VArMaxAbs", "Max Reactive Power Absorbed")
                .unit("var")
                .device_class("reactive_power")
                .state_class("measurement")
                .entity_category("diagnostic"),
        ];

        for sensor in nameplate_sensors {
            let ctx = DiscoveryContext {
                manufacturer,
                model,
                version,
                name: sensor.name,
                value_path: None,
                unit: sensor.unit,
                device_class: sensor.device_class,
                state_class: sensor.state_class,
                label: sensor.label,
                enabled_by_default: sensor.enabled_by_default,
                options: None,
                component: Some("sensor"),
                command_topic: None,
                entity_category: sensor.entity_category,
                state_topic: Some(nameplate_topic.clone()),
            };
            components.insert(
                sensor.name.to_string(),
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
            "stat_t": self.inverter_topic(serial),
            "qos": 1,
        });

        messages.push(MqttMessage::Publish {
            topic,
            payload: serde_json::to_vec(&payload)
                .map(bytes::Bytes::from)
                .unwrap_or_else(|_| bytes::Bytes::new()),
            retain: true,
        });

        messages
    }

    fn modern_component_payload(&self, serial: &str, ctx: &DiscoveryContext) -> serde_json::Value {
        let mut payload = json!({
            "p": ctx.component.unwrap_or("sensor"),
            "name": ctx.label,
            "uniq_id": format!("{}_{}_{}", self.topic_prefix, serial, ctx.name),
            "val_tpl": format!(
                "{{{{ value_json.{} }}}}",
                ctx.value_path.as_deref().unwrap_or(ctx.name)
            ),
            "en": ctx.enabled_by_default,
        });

        if let Some(state_topic) = &ctx.state_topic {
            payload["stat_t"] = json!(state_topic);
        }

        if let Some(cmd_topic) = &ctx.command_topic {
            payload["cmd_t"] = json!(cmd_topic);
        }

        if ctx.component == Some("switch") {
            payload["pl_on"] = json!(true);
            payload["pl_off"] = json!(false);
        }

        if let Some(unit) = ctx.unit {
            payload["unit_of_meas"] = json!(unit);
        }
        if let Some(dc) = ctx.device_class {
            payload["dev_cla"] = json!(dc);
        }
        if let Some(sc) = ctx.state_class {
            payload["stat_cla"] = json!(sc);
        }
        if let Some(options) = &ctx.options {
            payload["ops"] = json!(options);
        }
        if let Some(ec) = ctx.entity_category {
            payload["ent_cat"] = json!(ec);
        }

        payload
    }

    fn collect_sensor_definitions(&self, model_id: Option<u16>) -> Vec<SensorDefinition> {
        let mut sensors = vec![
            SensorDefinition::new("W", "Power")
                .unit("W")
                .device_class("power")
                .state_class("measurement"),
            SensorDefinition::new("WH", "Energy")
                .unit("Wh")
                .device_class("energy")
                .state_class("total_increasing"),
            SensorDefinition::new("Hz", "Frequency")
                .unit("Hz")
                .device_class("frequency")
                .state_class("measurement"),
            SensorDefinition::new("TmpCab", "Cabinet Temperature")
                .unit("°C")
                .device_class("temperature")
                .state_class("measurement"),
            SensorDefinition::new("TmpSnk", "Heat Sink Temperature")
                .unit("°C")
                .device_class("temperature")
                .state_class("measurement"),
            SensorDefinition::new("PF", "Power Factor")
                .device_class("power_factor")
                .state_class("measurement"),
            SensorDefinition::new("St", "Status").device_class("enum"),
        ];

        if let Some(id) = model_id {
            sensors.push(
                SensorDefinition::new("PhVphA", "Voltage Phase A")
                    .unit("V")
                    .device_class("voltage")
                    .state_class("measurement"),
            );
            sensors.push(
                SensorDefinition::new("AphA", "Current Phase A")
                    .unit("A")
                    .device_class("current")
                    .state_class("measurement"),
            );

            if id == 701 {
                sensors.extend(vec![
                    SensorDefinition::new("TmpAmb", "Ambient Temperature")
                        .unit("°C")
                        .device_class("temperature")
                        .state_class("measurement"),
                    SensorDefinition::new("TmpSw", "IGBT/MOSFET Temperature")
                        .unit("°C")
                        .device_class("temperature")
                        .state_class("measurement"),
                    SensorDefinition::new("Alrm", "Alarms"),
                    SensorDefinition::new("MnAlrmInfo", "Manufacturer Alarm Info"),
                    SensorDefinition::new("DERMode", "DER Operational Mode"),
                    SensorDefinition::new("ThrotPct", "Throttling Percentage")
                        .unit("%")
                        .state_class("measurement"),
                    SensorDefinition::new("ThrotSrc", "Throttling Source"),
                ]);
            }

            if matches!(id, 102 | 103 | 112 | 113 | 701) {
                sensors.push(
                    SensorDefinition::new("PhVphB", "Voltage Phase B")
                        .unit("V")
                        .device_class("voltage")
                        .state_class("measurement"),
                );
                sensors.push(
                    SensorDefinition::new("AphB", "Current Phase B")
                        .unit("A")
                        .device_class("current")
                        .state_class("measurement"),
                );
            }

            if matches!(id, 103 | 113 | 701) {
                sensors.push(
                    SensorDefinition::new("PhVphC", "Voltage Phase C")
                        .unit("V")
                        .device_class("voltage")
                        .state_class("measurement"),
                );
                sensors.push(
                    SensorDefinition::new("AphC", "Current Phase C")
                        .unit("A")
                        .device_class("current")
                        .state_class("measurement"),
                );
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

    pub fn discovery_message(&self, serial: &str, ctx: &DiscoveryContext) -> (String, bytes::Bytes) {
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

        (
            topic,
            serde_json::to_vec(&payload)
                .map(bytes::Bytes::from)
                .unwrap_or_else(|_| bytes::Bytes::new()),
        )
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
        assert_eq!(body["stat_t"], "solar/inverter/SN123");

        let cmps = body["cmps"].as_object().unwrap();

        // W should inherit state_topic
        assert!(cmps["W"]["stat_t"].is_null());

        // WMax (nameplate) should override state_topic
        assert_eq!(cmps["WMax"]["stat_t"], "solar/inverter/SN123/nameplate");
        assert_eq!(cmps["WMax"]["ent_cat"], "diagnostic");

        // Temperature sensors should be diagnostic
        assert_eq!(cmps["TmpCab"]["ent_cat"], "diagnostic");

        // Check unique_id format
        assert_eq!(cmps["W"]["uniq_id"], "solar_SN123_W");
    }

    #[test]
    fn test_wmaxlimpct_no_device_class() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let msgs = ha.generate_discovery_messages(
            "SN123",
            "Brand",
            "ModelX",
            Some("v1.0"),
            Some(101),
            crate::models::ActiveControlModel::Model704 { base_addr: 40000 },
        );

        let modern_msg = &msgs[0];
        let MqttMessage::Publish { payload, .. } = modern_msg;
        let body: serde_json::Value = serde_json::from_slice(payload).unwrap();
        let cmps = body["cmps"].as_object().unwrap();

        let wmaxlimpct = &cmps["WMaxLimPct"];
        assert!(
            wmaxlimpct.get("device_class").is_none(),
            "WMaxLimPct should not have a device_class"
        );
    }
}
