use super::*;
use crate::homeassistant::HomeAssistantIntegration;
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
        MqttMessage::Publish { topic, payload, .. } => {
            (topic, String::from_utf8(payload.to_vec()).unwrap())
        }
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
