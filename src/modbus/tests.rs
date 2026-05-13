use super::*;
use crate::homeassistant::{DiscoveryContext, HomeAssistantIntegration};
use chrono::Utc;
use tokio::sync::mpsc;
use tracing_test::traced_test;

fn test_task() -> ConnectionTask {
    let (tx, _) = mpsc::channel(1);
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());
    let (_, cmd_rx) = tokio::sync::broadcast::channel(1);
    ConnectionTask {
        config: ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: "127.0.0.1:502".to_string(),
                tls: false,
                ca_path: None,
                cert_path: None,
                key_path: None,
            },
            devices: vec![],
            keep_alive_interval: None,
        },
        mqtt_tx: tx,
        ha: HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string()),
        ha_enabled: true,
        token: CancellationToken::new(),
        root_cert_store,
        cmd_rx,
    }
}

#[test]
#[traced_test]
fn test_topics() {
    let task = test_task();
    assert_eq!(task.ha.inverter_topic("SN123"), "solar/inverter/SN123");

    let status_topic = task.ha.status_topic("SN123");
    let msg = task
        .ha
        .generate_status_message(status_topic, "OK", None, None);
    let (topic, payload) = match msg {
        MqttMessage::Publish { topic, payload, .. } => (topic, String::from_utf8(payload.to_vec()).unwrap()),
    };
    assert_eq!(topic, "solar/inverter/SN123/status");
    assert!(payload.contains("\"timestamp\":null"));

    let now = Utc::now();
    let status_topic = task.ha.status_topic("SN123");
    let msg_with_ts = task
        .ha
        .generate_status_message(status_topic, "OK", None, Some(&now));
    let payload = match msg_with_ts {
        MqttMessage::Publish { payload, .. } => String::from_utf8(payload.to_vec()).unwrap(),
    };
    assert!(payload.contains(&now.to_rfc3339()));
}

#[test]
#[traced_test]
fn test_discovery_message() {
    let task = test_task();
    let ctx = DiscoveryContext {
        manufacturer: "Brand",
        model: "ModelX",
        version: Some("1.2.3"),
        name: "W",
        value_path: None,
        unit: Some("W"),
        device_class: Some("power"),
        state_class: Some("measurement"),
        label: "Power",
        enabled_by_default: true,
        options: None,
        component: None,
        command_topic: None,
        entity_category: None,
        state_topic: None,
    };
    let (topic, payload_bytes) = task.ha.discovery_message("SN123", &ctx);
    let payload = String::from_utf8(payload_bytes.to_vec()).unwrap();

    assert_eq!(topic, "homeassistant/sensor/SN123/W/config");
    assert!(payload.contains("\"state_topic\":\"solar/inverter/SN123\""));
    assert!(payload.contains("\"value_template\":\"{{ value_json.W }}\""));
    assert!(payload.contains("\"unique_id\":\"solar_SN123_W\""));
    assert!(payload.contains("\"manufacturer\":\"Brand\""));
    assert!(payload.contains("\"model\":\"ModelX\""));
    assert!(payload.contains("\"sw_version\":\"1.2.3\""));
    assert!(payload.contains("\"state_class\":\"measurement\""));
    assert!(payload.contains("\"force_update\":true"));
    assert!(payload.contains("\"enabled_by_default\":true"));
}

#[test]
#[traced_test]
fn test_discovery_message_enum() {
    let task = test_task();
    let ctx = DiscoveryContext {
        manufacturer: "Brand",
        model: "ModelX",
        version: None,
        name: "St",
        value_path: None,
        unit: None,
        device_class: Some("enum"),
        state_class: None,
        label: "Status",
        enabled_by_default: true,
        options: Some(vec!["OFF", "ON"]),
        component: None,
        command_topic: None,
        entity_category: None,
        state_topic: None,
    };
    let (topic, payload_bytes) = task.ha.discovery_message("SN123", &ctx);
    let payload = String::from_utf8(payload_bytes.to_vec()).unwrap();

    assert_eq!(topic, "homeassistant/sensor/SN123/St/config");
    assert!(payload.contains("\"device_class\":\"enum\""));
    assert!(payload.contains("\"value_template\":\"{{ value_json.St }}\""));
    assert!(payload.contains("\"options\":[\"OFF\",\"ON\"]"));
    assert!(!payload.contains("\"sw_version\""));
    assert!(!payload.contains("\"unit_of_measurement\""));
    assert!(!payload.contains("\"state_class\""));
}
