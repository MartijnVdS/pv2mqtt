// SPDX-License-Identifier: Apache-2.0

use crate::error::Pv2MqttError;
use crate::mqtt::MqttMessage;
use chrono::{DateTime, Utc};
use serde::{Serialize, Serializer};
use std::sync::Arc;

pub struct HomeAssistantIntegration {
    topic_prefix: String,
    ha_prefix: String,
    status_buffer: bytes::BytesMut,
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
    pub command_topic: Option<Arc<str>>,
    pub entity_category: Option<&'static str>,
    pub state_topic: Option<Arc<str>>,
}

#[derive(Serialize)]
struct StatusPayload<'a> {
    #[serde(serialize_with = "serialize_opt_datetime")]
    timestamp: Option<&'a DateTime<Utc>>,
    status: &'a str,
    #[serde(serialize_with = "serialize_opt_display")]
    error: Option<&'a Pv2MqttError>,
    error_category: Option<&'static str>,
}

fn serialize_opt_datetime<S>(dt: &Option<&DateTime<Utc>>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match dt {
        Some(dt) => s.serialize_str(&dt.to_rfc3339()),
        None => s.serialize_none(),
    }
}

fn serialize_opt_display<S, T>(val: &Option<&T>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: std::fmt::Display,
{
    match val {
        Some(v) => s.collect_str(v),
        None => s.serialize_none(),
    }
}

#[derive(Serialize)]
struct HaDevice<'a> {
    ids: Vec<&'a str>,
    name: String,
    mf: &'a str,
    mdl: &'a str,
    #[serde(rename = "sw", skip_serializing_if = "Option::is_none")]
    sw_version: Option<&'a str>,
}

#[derive(Serialize)]
struct HaOrigin {
    name: &'static str,
    sw: &'static str,
}

#[derive(Serialize)]
struct ModernComponentPayload {
    #[serde(rename = "p")]
    component: String,
    name: String,
    #[serde(rename = "uniq_id")]
    unique_id: String,
    #[serde(rename = "val_tpl")]
    value_template: String,
    #[serde(rename = "en")]
    enabled: bool,
    #[serde(rename = "stat_t", skip_serializing_if = "Option::is_none")]
    state_topic: Option<Arc<str>>,
    #[serde(rename = "cmd_t", skip_serializing_if = "Option::is_none")]
    command_topic: Option<Arc<str>>,
    #[serde(rename = "pl_on", skip_serializing_if = "Option::is_none")]
    payload_on: Option<bool>,
    #[serde(rename = "pl_off", skip_serializing_if = "Option::is_none")]
    payload_off: Option<bool>,
    #[serde(rename = "unit_of_meas", skip_serializing_if = "Option::is_none")]
    unit_of_measurement: Option<&'static str>,
    #[serde(rename = "dev_cla", skip_serializing_if = "Option::is_none")]
    device_class: Option<&'static str>,
    #[serde(rename = "stat_cla", skip_serializing_if = "Option::is_none")]
    state_class: Option<&'static str>,
    #[serde(rename = "ops", skip_serializing_if = "Option::is_none")]
    options: Option<Vec<&'static str>>,
    #[serde(rename = "ent_cat", skip_serializing_if = "Option::is_none")]
    entity_category: Option<&'static str>,
}

#[derive(Serialize)]
struct ModernDiscoveryPayload<'a> {
    dev: HaDevice<'a>,
    o: HaOrigin,
    cmps: std::collections::BTreeMap<String, ModernComponentPayload>,
    stat_t: Arc<str>,
    qos: u8,
}

impl HomeAssistantIntegration {
    pub fn new(topic_prefix: String, ha_prefix: String) -> Self {
        Self {
            topic_prefix,
            ha_prefix,
            status_buffer: bytes::BytesMut::with_capacity(256),
        }
    }

    pub fn inverter_topic(&self, serial: &str) -> Arc<str> {
        format!("{}/inverter/{}", self.topic_prefix, serial).into()
    }

    pub fn status_topic(&self, serial: &str) -> Arc<str> {
        format!("{}/inverter/{}/status", self.topic_prefix, serial).into()
    }

    pub fn discovery_topic(&self, serial: &str) -> Arc<str> {
        format!("{}/device/{}/config", self.ha_prefix, serial).into()
    }

    pub fn nameplate_topic(&self, serial: &str) -> Arc<str> {
        format!("{}/inverter/{}/nameplate", self.topic_prefix, serial).into()
    }

    pub fn generate_status_message(
        &mut self,
        topic: Arc<str>,
        status: &str,
        error: Option<&Pv2MqttError>,
        last_success: Option<&DateTime<Utc>>,
    ) -> MqttMessage {
        use bytes::BufMut;
        self.status_buffer.clear();
        let payload = match serde_json::to_writer(
            (&mut self.status_buffer).writer(),
            &StatusPayload {
                timestamp: last_success,
                status,
                error,
                error_category: error.as_ref().map(|e| e.category()),
            },
        ) {
            Ok(_) => self.status_buffer.split().freeze(),
            Err(_) => bytes::Bytes::new(),
        };

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
        let mut components = std::collections::BTreeMap::new();

        self.add_standard_sensors(
            serial,
            manufacturer,
            model,
            version,
            model_id,
            &mut components,
        );
        self.add_binary_sensors(
            serial,
            manufacturer,
            model,
            version,
            model_id,
            &mut components,
        );
        self.add_controls(
            serial,
            manufacturer,
            model,
            version,
            active_control,
            &mut components,
        );
        self.add_nameplate_sensors(serial, manufacturer, model, version, &mut components);

        let payload = ModernDiscoveryPayload {
            dev: HaDevice {
                ids: vec![serial],
                name: format!("Inverter {}", serial),
                mf: manufacturer,
                mdl: model,
                sw_version: version,
            },
            o: HaOrigin {
                name: "pv2mqtt",
                sw: env!("CARGO_PKG_VERSION"),
            },
            cmps: components,
            stat_t: self.inverter_topic(serial),
            qos: 1,
        };

        vec![MqttMessage::Publish {
            topic: self.discovery_topic(serial),
            payload: serde_json::to_vec(&payload)
                .map(bytes::Bytes::from)
                .unwrap_or_else(|_| bytes::Bytes::new()),
            retain: true,
        }]
    }

    fn add_standard_sensors(
        &self,
        serial: &str,
        manufacturer: &str,
        model: &str,
        version: Option<&str>,
        model_id: Option<u16>,
        components: &mut std::collections::BTreeMap<String, ModernComponentPayload>,
    ) {
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
    }

    fn add_binary_sensors(
        &self,
        serial: &str,
        manufacturer: &str,
        model: &str,
        version: Option<&str>,
        model_id: Option<u16>,
        components: &mut std::collections::BTreeMap<String, ModernComponentPayload>,
    ) {
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
    }

    fn add_controls(
        &self,
        serial: &str,
        manufacturer: &str,
        model: &str,
        version: Option<&str>,
        active_control: crate::models::ActiveControlModel,
        components: &mut std::collections::BTreeMap<String, ModernComponentPayload>,
    ) {
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
                command_topic: Some(
                    format!(
                        "{}/inverter/{}/set/{}",
                        self.topic_prefix, serial, control.name
                    )
                    .into(),
                ),

                entity_category: None,
                state_topic: None,
            };
            components.insert(
                control.name.to_string(),
                self.modern_component_payload(serial, &ctx),
            );
        }
    }

    fn add_nameplate_sensors(
        &self,
        serial: &str,
        manufacturer: &str,
        model: &str,
        version: Option<&str>,
        components: &mut std::collections::BTreeMap<String, ModernComponentPayload>,
    ) {
        let nameplate_topic = self.nameplate_topic(serial);
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
    }

    fn modern_component_payload(
        &self,
        serial: &str,
        ctx: &DiscoveryContext,
    ) -> ModernComponentPayload {
        ModernComponentPayload {
            component: ctx.component.unwrap_or("sensor").to_string(),
            name: ctx.label.to_string(),
            unique_id: format!("{}_{}_{}", self.topic_prefix, serial, ctx.name),
            value_template: format!(
                "{{{{ value_json.{} }}}}",
                ctx.value_path.as_deref().unwrap_or(ctx.name)
            ),
            enabled: ctx.enabled_by_default,
            state_topic: ctx.state_topic.clone(),
            command_topic: ctx.command_topic.clone(),
            payload_on: if ctx.component == Some("switch") {
                Some(true)
            } else {
                None
            },
            payload_off: if ctx.component == Some("switch") {
                Some(false)
            } else {
                None
            },
            unit_of_measurement: ctx.unit,
            device_class: ctx.device_class,
            state_class: ctx.state_class,
            options: ctx.options.clone(),
            entity_category: ctx.entity_category,
        }
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

        let Some(id) = model_id else {
            return sensors;
        };

        // Phase A is common to all models if model_id is present
        sensors.extend(vec![
            SensorDefinition::new("PhVphA", "Voltage Phase A")
                .unit("V")
                .device_class("voltage")
                .state_class("measurement"),
            SensorDefinition::new("AphA", "Current Phase A")
                .unit("A")
                .device_class("current")
                .state_class("measurement"),
        ]);

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
            sensors.extend(vec![
                SensorDefinition::new("PhVphB", "Voltage Phase B")
                    .unit("V")
                    .device_class("voltage")
                    .state_class("measurement"),
                SensorDefinition::new("AphB", "Current Phase B")
                    .unit("A")
                    .device_class("current")
                    .state_class("measurement"),
            ]);
        }

        if matches!(id, 103 | 113 | 701) {
            sensors.extend(vec![
                SensorDefinition::new("PhVphC", "Voltage Phase C")
                    .unit("V")
                    .device_class("voltage")
                    .state_class("measurement"),
                SensorDefinition::new("AphC", "Current Phase C")
                    .unit("A")
                    .device_class("current")
                    .state_class("measurement"),
            ]);
        }

        sensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_helpers() {
        let ha = HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string());
        let serial = "SN123";

        assert_eq!(&*ha.inverter_topic(serial), "solar/inverter/SN123");
        assert_eq!(&*ha.status_topic(serial), "solar/inverter/SN123/status");
        assert_eq!(
            &*ha.discovery_topic(serial),
            "homeassistant/device/SN123/config"
        );
        assert_eq!(
            &*ha.nameplate_topic(serial),
            "solar/inverter/SN123/nameplate"
        );
    }

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
        assert_eq!(&**topic, "homeassistant/device/SN123/config");
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
        assert!(body["cmps"].as_object().unwrap().contains_key("Alrm"));
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
