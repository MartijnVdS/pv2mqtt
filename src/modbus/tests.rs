// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::modbus::connection::MockInverter;
use crate::models::ActiveControlModel;
use chrono::Utc;
use tokio::sync::mpsc;
use tracing_test::traced_test;

fn test_task() -> ConnectionTask<MockInverter> {
    let (tx, _) = mpsc::channel(1);
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());
    let (_, cmd_rx) = tokio::sync::broadcast::channel(1);
    ConnectionTask::<MockInverter>::new(
        ConnectionConfig {
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
        tx,
        &crate::config::MqttConfig {
            url: "mqtt://localhost".to_string(),
            client_id: "test".to_string(),
            topic_prefix: "solar".to_string(),
            ha_prefix: "homeassistant".to_string(),
            ha_enabled: true,
            ca_path: None,
            cert_path: None,
            key_path: None,
        },
        CancellationToken::new(),
        root_cert_store,
        cmd_rx,
    )
}

#[test]
#[traced_test]
fn test_topics() {
    let task = test_task();
    assert_eq!(&*task.ha.inverter_topic("SN123"), "solar/inverter/SN123");

    let status_topic = task.ha.status_topic("SN123");
    let mut ha = task.ha;
    let msg = ha.generate_status_message(status_topic, "OK", None, None);
    let (topic, payload) = match msg {
        MqttMessage::Publish { topic, payload, .. } => {
            (topic, String::from_utf8(payload.to_vec()).unwrap())
        }
    };
    assert_eq!(&*topic, "solar/inverter/SN123/status");
    assert!(payload.contains("\"timestamp\":null"));

    let now = Utc::now();
    let status_topic = ha.status_topic("SN123");
    let msg_with_ts = ha.generate_status_message(status_topic, "OK", None, Some(&now));
    let payload = match msg_with_ts {
        MqttMessage::Publish { payload, .. } => String::from_utf8(payload.to_vec()).unwrap(),
    };
    assert!(payload.contains(&now.to_rfc3339()));
}

#[test]
fn test_calculate_next_wakeup() {
    let _task = test_task();
    let keep_alive_interval = 30;
    let keep_alive_duration = Duration::from_secs(keep_alive_interval);

    let now = Instant::now();
    let last_activity = now;

    // 1. Idle fallback (no devices)
    let wakeup = ConnectionTask::<MockInverter>::calculate_next_wakeup(
        &[],
        last_activity,
        keep_alive_duration,
        keep_alive_interval,
    );
    assert!(wakeup >= now + Duration::from_secs(DEFAULT_IDLE_SLEEP_SECS));

    // 2. Poll priority (without device, it still calculates next poll)
    let mut device = DeviceState::<MockInverter>::new(crate::config::DeviceConfig {
        unit_id: 1,
        interval: 60,
        preferred_model: None,
        enable_controls: false,
    });
    device.last_poll = Some(now); // Next poll in 60s

    // Test priority logic by setting a mock device
    device.device = Some(Arc::new(MockInverter {
        slave_id: 1,
        metadata: DeviceMetadata {
            serial: "S1".into(),
            manufacturer: "M1".into(),
            model: "MDL1".into(),
            version: None,
            supported_model: 103,
            active_control: ActiveControlModel::None,
        },
        nameplate: None,
        model_ids: vec![1, 103],
    }));

    let mut d_poll_soon = device.clone();
    d_poll_soon.last_poll = Some(now - Duration::from_secs(50)); // Due in 10s
    let wakeup = ConnectionTask::<MockInverter>::calculate_next_wakeup(
        &[d_poll_soon.clone()],
        last_activity,
        keep_alive_duration,
        keep_alive_interval,
    );
    assert_eq!(
        wakeup,
        d_poll_soon.last_poll.unwrap() + Duration::from_secs(60)
    );

    // 3. Overdue polls
    let mut d_overdue = device.clone();
    d_overdue.last_poll = Some(now - Duration::from_secs(70)); // Due 10s ago
    let wakeup = ConnectionTask::<MockInverter>::calculate_next_wakeup(
        &[d_overdue],
        last_activity,
        keep_alive_duration,
        keep_alive_interval,
    );
    assert!(wakeup <= Instant::now());

    // 4. Minimum across multiple devices
    let mut d1 = device.clone();
    d1.last_poll = Some(now - Duration::from_secs(40)); // Due in 20s
    let mut d2 = device.clone();
    d2.last_poll = Some(now - Duration::from_secs(55)); // Due in 5s
    let wakeup = ConnectionTask::<MockInverter>::calculate_next_wakeup(
        &[d1, d2.clone()],
        last_activity,
        keep_alive_duration,
        keep_alive_interval,
    );
    assert_eq!(wakeup, d2.last_poll.unwrap() + Duration::from_secs(60));
}
