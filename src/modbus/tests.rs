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
    let mut task = test_task();
    assert_eq!(&*task.ha.inverter_topic("SN123"), "solar/inverter/SN123");

    let status_topic = task.ha.status_topic("SN123");
    let msg = task
        .ha
        .generate_status_message(status_topic, "OK", None, None);
    let (topic, payload) = match msg {
        MqttMessage::Publish { topic, payload, .. } => {
            (topic, String::from_utf8(payload.to_vec()).unwrap())
        }
    };
    assert_eq!(&*topic, "solar/inverter/SN123/status");
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
fn test_calculate_next_wakeup() {
    let task = test_task();
    let keep_alive_interval = 30;
    let keep_alive_duration = Duration::from_secs(keep_alive_interval);

    let now = Instant::now();
    let last_activity = now;

    // 1. Idle fallback (no devices)
    let wakeup =
        task.calculate_next_wakeup(&[], last_activity, keep_alive_duration, keep_alive_interval);
    assert!(wakeup >= now + Duration::from_secs(DEFAULT_IDLE_SLEEP_SECS));

    // 2. Keep-alive priority
    let mut device = DeviceState::new(crate::config::DeviceConfig {
        unit_id: 1,
        interval: 60,
        preferred_model: None,
        enable_controls: false,
    });
    device.last_poll = Some(now); // Next poll in 60s
    // To make any(|d| d.device.is_some()) true, we need a "device"
    // Since we can't easily make a real AsyncDevice, we'll just test the logic that depends on it
    // by manually setting it if possible, but AsyncDevice is hard to mock.
    // Wait, the logic is:
    // if keep_alive_interval > 0 && devices.iter().any(|d| d.device.is_some())

    // 2. Poll priority (without device, it still calculates next poll)
    let mut d_poll_soon = device.clone();
    d_poll_soon.last_poll = Some(now - Duration::from_secs(50)); // Due in 10s
    let wakeup = task.calculate_next_wakeup(
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
    let wakeup = task.calculate_next_wakeup(
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
    let wakeup = task.calculate_next_wakeup(
        &[d1, d2.clone()],
        last_activity,
        keep_alive_duration,
        keep_alive_interval,
    );
    assert_eq!(wakeup, d2.last_poll.unwrap() + Duration::from_secs(60));
}
