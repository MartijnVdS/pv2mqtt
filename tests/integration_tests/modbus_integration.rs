// SPDX-License-Identifier: Apache-2.0

use crate::common::modbus_server;
use pv2mqtt::config::{ConnectionConfig, DeviceConfig, ModbusConfig};
use pv2mqtt::homeassistant::HomeAssistantIntegration;
use pv2mqtt::modbus::{
    ConnectionTask, M123_CONN_OFFSET, M123_WMAX_LIM_PCT_SF_OFFSET, M704_WMAX_LIM_PCT_OFFSET,
    M704_WMAX_LIM_PCT_SF_OFFSET, RECONNECT_TIMEOUT_SECS,
};
use pv2mqtt::mqtt::MqttMessage;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;
use tracing_test::traced_test;

fn setup_mock_sunspec_registers(r: &mut [u16]) -> usize {
    r[40000] = 0x5375; // 'Su'
    r[40001] = 0x6e53; // 'nS'

    // Model 1 (Common)
    r[40002] = 1; // Model ID
    r[40003] = 66; // Length
    // Pre-fill with spaces
    for i in 0..66 {
        r[40004 + i] = 0x2020;
    }
    // Manufacturer (Mn) starts at offset 0 of Model 1 data (r[40004])
    r[40004] = 0x4272; // 'Br'
    r[40005] = 0x616e; // 'an'
    r[40006] = 0x6420; // 'd '
    // Model (Md) starts at offset 16 of Model 1 data (r[40004+16=40020])
    r[40020] = 0x4d6f; // 'Mo'
    r[40021] = 0x6465; // 'de'
    r[40022] = 0x6c20; // 'l '
    // Serial number (SN) starts at offset 50 (40004 + 50 = 40054)
    r[40054] = 0x534e; // 'SN'
    r[40055] = 0x3132; // '12'
    r[40056] = 0x3334; // '34'

    // Model 103 (Three Phase Inverter) at 40070 (40002 + 66 + 2)
    r[40070] = 103; // Model ID
    r[40071] = 50; // Length
    let m103_base = 40072;
    r[m103_base + 12] = 1000;
    r[m103_base + 13] = 0;

    // Return the address for the next model (40070 + 50 + 2)
    40122
}

#[tokio::test]
#[traced_test]
async fn test_reconnection_logic() {
    // Use tokio::time::pause() to advance time and skip the "real" timeout
    tokio::time::pause();

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let handle_mock = modbus_server::start_mock_server(addr).await;
    let addr_str = handle_mock.addr.to_string();

    let (tx, _rx) = mpsc::channel(1);
    let token = CancellationToken::new();
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());

    let (_, cmd_rx) = tokio::sync::broadcast::channel(1);
    let task = ConnectionTask {
        config: ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: addr_str,
                tls: false,
                ca_path: None,
                cert_path: None,
                key_path: None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 1,
                enable_controls: true,
                preferred_model: None,
            }],
            keep_alive_interval: None,
        },
        mqtt_tx: tx,
        ha: HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string()),
        ha_enabled: true,
        token: token.clone(),
        root_cert_store,
        cmd_rx,
    };

    let token_clone = token.clone();
    let task_handle = tokio::spawn(
        async move { task.run_internal().await }.instrument(tracing::info_span!("run")),
    );

    // Wait for the first connection
    handle_mock.notify.notified().await;
    assert_eq!(handle_mock.connections.load(Ordering::SeqCst), 1);

    // Advance time. Discovery will fail (no SunSpec marker), but it should NOT reconnect.
    tokio::time::advance(Duration::from_secs(1)).await;

    // Wait some time and check that connections is still 1
    tokio::time::advance(Duration::from_secs(RECONNECT_TIMEOUT_SECS * 2)).await;
    assert_eq!(handle_mock.connections.load(Ordering::SeqCst), 1);

    token_clone.cancel();
    let result = task_handle.await.unwrap();
    assert!(result.is_ok()); // run_internal returns Ok(()) on cancellation
}

#[tokio::test]
#[traced_test]
async fn test_successful_poll_logic() {
    // Use tokio time travel
    tokio::time::pause();

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let handle_mock = modbus_server::start_mock_server(addr).await;
    let addr_str = handle_mock.addr.to_string();
    let regs = handle_mock.registers;

    // Setup SunSpec discovery registers
    {
        let mut r = regs.lock().unwrap();
        let next_addr = setup_mock_sunspec_registers(&mut r);

        // St at offset 36 of Model 103
        r[40072 + 36] = 1; // OFF

        // End of models marker
        r[next_addr] = 0xFFFF;
        r[next_addr + 1] = 0;
    }

    let (tx, mut rx) = mpsc::channel(100);
    let token = CancellationToken::new();
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());

    let (_, cmd_rx) = tokio::sync::broadcast::channel(1);
    let task = ConnectionTask {
        config: ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: addr_str,
                tls: false,
                ca_path: None,
                cert_path: None,
                key_path: None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 1,
                enable_controls: true,
                preferred_model: None,
            }],
            keep_alive_interval: None,
        },
        mqtt_tx: tx,
        ha: HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string()),
        ha_enabled: true,
        token: token.clone(),
        root_cert_store,
        cmd_rx,
    };

    let token_clone = token.clone();
    let task_handle = tokio::spawn(
        async move { task.run_internal().await }.instrument(tracing::info_span!("run")),
    );

    // Collect messages by advancing time in small increments
    let mut messages = Vec::new();
    // Discovery (12) + cleanup for disabled controls (3) + at least one poll (2: Data + Status) = 17 messages
    let mut timeout_counter = 0;
    while messages.len() < 17 && timeout_counter < 500 {
        tokio::time::advance(Duration::from_millis(10)).await;
        while let Ok(msg) = rx.try_recv() {
            messages.push(msg);
        }
        timeout_counter += 1;
    }

    assert!(
        messages.len() >= 17,
        "Should have received at least 17 MQTT messages, got {}",
        messages.len()
    );

    // We expect discovery messages (12) + at least one poll (Data + Status)
    // Discovery messages have topics starting with homeassistant/
    let discovery_msgs: Vec<_> = messages
        .iter()
        .filter(|m| {
            let MqttMessage::Publish { topic, .. } = m;
            topic.starts_with("homeassistant/")
        })
        .collect();
    assert!(
        discovery_msgs.len() >= 12,
        "Should have several discovery messages"
    );

    // Verify one specific discovery message (e.g., Power sensor 'W')
    let w_discovery = discovery_msgs
        .iter()
        .find(|m| {
            let MqttMessage::Publish { topic, .. } = m;
            topic.contains("/W/")
        })
        .expect("Should have discovery for 'W'");

    let MqttMessage::Publish { topic, payload, .. } = w_discovery;
    assert_eq!(*topic, "homeassistant/sensor/SN1234/W/config");
    let payload_str = String::from_utf8_lossy(payload);
    assert!(payload_str.contains("\"manufacturer\":\"Brand\""));
    assert!(payload_str.contains("\"model\":\"Model\""));
    assert!(payload_str.contains("\"unique_id\":\"solar_SN1234_W\""));

    // Check for data messages
    let data_msgs: Vec<_> = messages
        .iter()
        .filter(|m| {
            let MqttMessage::Publish { topic, .. } = m;
            topic == "solar/inverter/SN1234"
        })
        .collect();
    assert!(!data_msgs.is_empty(), "Should have received poll data");

    let MqttMessage::Publish { payload, .. } = &data_msgs[0];
    let json: serde_json::Value = serde_json::from_slice(payload).unwrap();
    assert_eq!(json["W"], 1000.0);
    assert_eq!(json["St"], "OFF");

    // Check for status messages
    let status_msgs: Vec<_> = messages
        .iter()
        .filter(|m| {
            let MqttMessage::Publish { topic, .. } = m;
            topic == "solar/inverter/SN1234/status"
        })
        .collect();
    assert!(
        !status_msgs.is_empty(),
        "Should have received status updates"
    );

    let MqttMessage::Publish { payload, .. } = &status_msgs[0];
    let json: serde_json::Value = serde_json::from_slice(payload).unwrap();
    assert_eq!(json["status"], "OK");

    token_clone.cancel();
    let _ = task_handle.await.unwrap();
}

#[tokio::test]
#[traced_test]
async fn test_command_execution_logic() {
    // Use tokio time travel for deterministic command execution
    tokio::time::pause();

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let handle_mock = modbus_server::start_mock_server(addr).await;
    let addr_str = handle_mock.addr.to_string();
    let regs = handle_mock.registers;

    // Setup SunSpec discovery registers
    {
        let mut r = regs.lock().unwrap();
        let next_addr = setup_mock_sunspec_registers(&mut r);

        // Model 123 (Immediate Controls)
        r[next_addr] = 123; // Model ID
        r[next_addr + 1] = 24; // Length
        // Constants are data-relative, so we add 2 to match model-relative layout
        r[next_addr + (M123_WMAX_LIM_PCT_SF_OFFSET as usize + 2)] = 0xFFFE; // SF = -2 (as i16)

        // End of models marker
        r[next_addr + 26] = 0xFFFF;
        r[next_addr + 27] = 0;
    }

    let (tx, mut rx) = mpsc::channel(100);
    let token = CancellationToken::new();
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());

    let (cmd_tx, cmd_rx) = tokio::sync::broadcast::channel(1);
    let task = ConnectionTask {
        config: ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: addr_str,
                tls: false,
                ca_path: None,
                cert_path: None,
                key_path: None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 1,
                enable_controls: true,
                preferred_model: None,
            }],
            keep_alive_interval: None,
        },
        mqtt_tx: tx,
        ha: HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string()),
        ha_enabled: true,
        token: token.clone(),
        root_cert_store,
        cmd_rx,
    };

    let token_clone = token.clone();
    let task_handle = tokio::spawn(
        async move { task.run_internal().await }.instrument(tracing::info_span!("run")),
    );

    // Wait a bit for discovery to finish
    tokio::time::advance(Duration::from_millis(100)).await;

    // Discovery message is published after successful poll
    // We advance time to trigger the poll interval.
    let mut discovery_msg = None;
    let mut timeout_counter = 0;
    while discovery_msg.is_none() && timeout_counter < 500 {
        tokio::time::advance(Duration::from_millis(10)).await;
        if let Ok(msg) = rx.try_recv() {
            discovery_msg = Some(msg);
            break;
        }
        timeout_counter += 1;
    }
    discovery_msg.expect("Did not receive discovery message within timeout");

    // The data block starts at 40124 (next_addr + 2)
    let m123_data_base = 40124;

    // Send a command (WMaxLimPct)
    let cmd = pv2mqtt::commands::ModbusCommand {
        serial: "SN1234".to_string(),
        action: pv2mqtt::commands::ControlAction::WMaxLimPct(75.5),
    };
    cmd_tx.send(cmd).unwrap();

    // Advance time and verify write to data-relative offset
    let mut write_verified = false;
    for _ in 0..10 {
        tokio::time::advance(Duration::from_millis(10)).await;
        let r = regs.lock().unwrap();
        if r[m123_data_base + pv2mqtt::modbus::M123_WMAX_LIM_PCT_OFFSET as usize] == 7550 {
            write_verified = true;
            break;
        }
    }
    assert!(
        write_verified,
        "WMaxLimPct register write did not occur at expected offset"
    );

    // Send a Conn command
    let cmd = pv2mqtt::commands::ModbusCommand {
        serial: "SN1234".to_string(),
        action: pv2mqtt::commands::ControlAction::Conn(true),
    };
    cmd_tx.send(cmd).unwrap();

    let mut conn_verified = false;
    for _ in 0..100 {
        tokio::time::advance(Duration::from_millis(50)).await;
        let r = regs.lock().unwrap();
        if r[m123_data_base + M123_CONN_OFFSET as usize] == 1 {
            conn_verified = true;
            break;
        }
    }
    assert!(
        conn_verified,
        "Conn register write did not occur at expected offset"
    );

    token_clone.cancel();
    // Advance time to allow the cancellation to propagate and loop to exit
    tokio::time::advance(Duration::from_millis(100)).await;
    let _ = task_handle.await.unwrap();
}

#[tokio::test]
#[traced_test]
async fn test_command_ignored_when_controls_disabled() {
    tokio::time::pause();

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let handle_mock = modbus_server::start_mock_server(addr).await;
    let addr_str = handle_mock.addr.to_string();
    let regs = handle_mock.registers;

    {
        let mut r = regs.lock().unwrap();
        let next_addr = setup_mock_sunspec_registers(&mut r);
        r[next_addr] = 123;
        r[next_addr + 1] = 24;
        r[next_addr + 2 + 23] = 0xFFFE; // SF = -2
        r[next_addr + 2 + 24] = 0xFFFF;
    }

    let (tx, mut rx) = mpsc::channel(100);
    let token = CancellationToken::new();
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());

    let (cmd_tx, cmd_rx) = tokio::sync::broadcast::channel(1);
    let task = ConnectionTask {
        config: ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: addr_str,
                tls: false,
                ca_path: None,
                cert_path: None,
                key_path: None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 1,
                enable_controls: false, // DISABLED
                preferred_model: None,
            }],
            keep_alive_interval: None,
        },
        mqtt_tx: tx,
        ha: HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string()),
        ha_enabled: true,
        token: token.clone(),
        root_cert_store,
        cmd_rx,
    };

    let token_clone = token.clone();
    let task_handle = tokio::spawn(
        async move { task.run_internal().await }.instrument(tracing::info_span!("run")),
    );

    // Advance to finish discovery
    let mut messages = Vec::new();
    for _ in 0..100 {
        tokio::time::advance(Duration::from_millis(10)).await;
        while let Ok(msg) = rx.try_recv() {
            messages.push(msg);
        }
        if !messages.is_empty() {
            break;
        }
    }
    // Send a command
    let cmd = pv2mqtt::commands::ModbusCommand {
        serial: "SN1234".to_string(),
        action: pv2mqtt::commands::ControlAction::WMaxLimPct(75.5),
    };
    cmd_tx.send(cmd).unwrap();

    // Advance time to allow processing.
    // We use a loop to ensure handle_command has a chance to run.
    for _ in 0..10 {
        tokio::time::advance(Duration::from_millis(100)).await;
    }

    {
        let r = regs.lock().unwrap();
        // Register should still be 0, not 7550
        assert_eq!(r[40124 + 3], 0);
    }

    token_clone.cancel();
    tokio::time::advance(Duration::from_millis(100)).await;
    let _ = task_handle.await.unwrap();
}

#[tokio::test]
#[traced_test]
async fn test_discovery_disabled_logic() {
    tokio::time::pause();

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let handle_mock = modbus_server::start_mock_server(addr).await;
    let addr_str = handle_mock.addr.to_string();
    let regs = handle_mock.registers;

    // Setup SunSpec discovery registers
    {
        let mut r = regs.lock().unwrap();
        let next_addr = setup_mock_sunspec_registers(&mut r);
        r[next_addr] = 0xFFFF;
    }

    let (tx, mut rx) = mpsc::channel(100);
    let token = CancellationToken::new();
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());

    let (_, cmd_rx) = tokio::sync::broadcast::channel(1);
    let task = ConnectionTask {
        config: ConnectionConfig {
            name: "test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: addr_str,
                tls: false,
                ca_path: None,
                cert_path: None,
                key_path: None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 1,
                enable_controls: true,
                preferred_model: None,
            }],
            keep_alive_interval: None,
        },
        mqtt_tx: tx,
        ha: HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string()),
        ha_enabled: false, // DISABLED
        token: token.clone(),
        root_cert_store,
        cmd_rx,
    };

    let token_clone = token.clone();
    let task_handle = tokio::spawn(
        async move { task.run_internal().await }.instrument(tracing::info_span!("run")),
    );

    // Collect messages
    let mut messages = Vec::new();
    for _ in 0..100 {
        tokio::time::advance(Duration::from_millis(10)).await;
        while let Ok(msg) = rx.try_recv() {
            messages.push(msg);
        }
        // Wait until we get at least one non-discovery message (e.g. status)
        if messages.iter().any(|m| {
            let MqttMessage::Publish { topic, .. } = m;
            topic.ends_with("/status")
        }) {
            break;
        }
    }

    // Verify NO discovery messages were sent
    let discovery_msgs: Vec<_> = messages
        .iter()
        .filter(|m| {
            let MqttMessage::Publish { topic, .. } = m;
            topic.starts_with("homeassistant/")
        })
        .collect();

    assert!(
        discovery_msgs.is_empty(),
        "Discovery messages should NOT be sent when ha_enabled is false"
    );

    // Verify that data/status messages WERE sent
    assert!(
        !messages.is_empty(),
        "Some messages (data/status) should still be sent"
    );

    token_clone.cancel();
    tokio::time::advance(Duration::from_millis(100)).await;
    let _ = task_handle.await.unwrap();
}

#[tokio::test]
#[traced_test]
async fn test_model_704_command_execution_logic() {
    tokio::time::pause();

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let handle_mock = modbus_server::start_mock_server(addr).await;
    let addr_str = handle_mock.addr.to_string();
    let regs = handle_mock.registers;

    {
        let mut r = regs.lock().unwrap();
        let next_addr = setup_mock_sunspec_registers(&mut r);

        // Model 704 (DER AC Controls)
        r[next_addr] = 704;
        r[next_addr + 1] = 65;
        // SF = -2 at offset 54 (Relative to data block start at next_addr + 2)
        r[next_addr + 2 + M704_WMAX_LIM_PCT_SF_OFFSET as usize] = 0xFFFE;

        // End of models marker
        r[next_addr + 67] = 0xFFFF;
        r[next_addr + 68] = 0;
    }

    let (tx, mut rx) = mpsc::channel(100);
    let token = CancellationToken::new();
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());

    let (cmd_tx, cmd_rx) = tokio::sync::broadcast::channel(1);
    let task = ConnectionTask {
        config: ConnectionConfig {
            name: "test-704".to_string(),
            modbus: ModbusConfig::Tcp {
                address: addr_str,
                tls: false,
                ca_path: None,
                cert_path: None,
                key_path: None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 1,
                enable_controls: true,
                preferred_model: None,
            }],
            keep_alive_interval: None,
        },
        mqtt_tx: tx,
        ha: HomeAssistantIntegration::new("solar".to_string(), "homeassistant".to_string()),
        ha_enabled: true,
        token: token.clone(),
        root_cert_store,
        cmd_rx,
    };

    let token_clone = token.clone();
    let task_handle = tokio::spawn(
        async move { task.run_internal().await }.instrument(tracing::info_span!("run")),
    );

    tokio::time::advance(Duration::from_millis(100)).await;
    let mut messages = Vec::new();
    for _ in 0..100 {
        tokio::time::advance(Duration::from_millis(10)).await;
        while let Ok(msg) = rx.try_recv() {
            messages.push(msg);
        }
        if !messages.is_empty() {
            break;
        }
    }

    // Send a command (WMaxLimPct)
    let cmd = pv2mqtt::commands::ModbusCommand {
        serial: "SN1234".to_string(),
        action: pv2mqtt::commands::ControlAction::WMaxLimPct(80.0),
    };
    cmd_tx.send(cmd).unwrap();

    let mut write_verified = false;
    for _ in 0..10 {
        tokio::time::advance(Duration::from_millis(100)).await;
        let r = regs.lock().unwrap();
        // Base addr for 704 data is 40124 (next_addr + 2)
        if r[40124 + M704_WMAX_LIM_PCT_OFFSET as usize] == 8000 {
            write_verified = true;
            break;
        }
    }
    assert!(write_verified, "Model 704 WMaxLimPct write did not occur");

    token_clone.cancel();
    tokio::time::advance(Duration::from_millis(100)).await;
    let _ = task_handle.await.unwrap();
}
