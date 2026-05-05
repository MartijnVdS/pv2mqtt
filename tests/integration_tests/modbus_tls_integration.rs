// SPDX-License-Identifier: Apache-2.0

use crate::common::modbus_server;
use pv2mqtt::config::{ConnectionConfig, DeviceConfig, ModbusConfig};
use pv2mqtt::homeassistant::HomeAssistantIntegration;
use pv2mqtt::modbus::ConnectionTask;
use rcgen::generate_simple_self_signed;
use std::fs;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;
use tracing_test::traced_test;

#[tokio::test]
#[traced_test]
async fn test_tls_handshake_failure() {
    // Generate a self-signed certificate for the server
    let subject_alt_names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    let certified_key = generate_simple_self_signed(subject_alt_names).unwrap();
    let cert_der = certified_key.cert.der().to_vec();
    let key_der = certified_key.key_pair.serialize_der();

    // Create a TlsAcceptor for the mock server
    let server_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(
            vec![rustls::pki_types::CertificateDer::from(cert_der)],
            rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
                key_der,
            )),
        )
        .unwrap();
    let acceptor = TlsAcceptor::from(Arc::new(server_config));

    // Start the TLS mock server
    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let handle_mock = modbus_server::start_tls_mock_server(addr, acceptor).await;
    let addr_str = handle_mock.addr.to_string();

    // Set up paths for "untrusted" CA (we won't actually use the correct CA on the client)
    let temp_dir = "tests/temp_certs";
    fs::create_dir_all(temp_dir).unwrap();
    let ca_path = format!("{}/untrusted_ca.crc", temp_dir);

    // Generate a DIFFERENT CA to ensure handshake failure due to untrusted cert
    let other_key = generate_simple_self_signed(vec!["other".to_string()]).unwrap();
    fs::write(&ca_path, other_key.cert.pem()).unwrap();

    let (tx, _rx) = mpsc::channel(1);
    let token = CancellationToken::new();
    let root_cert_store = Arc::new(rustls::RootCertStore::empty());

    let (_, cmd_rx) = tokio::sync::broadcast::channel(1);
    let task = ConnectionTask {
        config: ConnectionConfig {
            name: "tls_test".to_string(),
            modbus: ModbusConfig::Tcp {
                address: addr_str,
                tls: true,
                ca_path: Some(ca_path.clone()),
                cert_path: None,
                key_path: None,
            },
            devices: vec![DeviceConfig {
                unit_id: 1,
                interval: 1,
                enable_controls: false,
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

    let _task_handle = tokio::spawn(
        async move { task.run_internal().await }.instrument(tracing::info_span!("run")),
    );

    // Use virtual time to avoid real-world sleeps
    tokio::time::pause();

    // We expect a "TLS handshake failed" or similar error in the logs
    let mut failure_detected = false;
    for _ in 0..50 {
        tokio::time::advance(std::time::Duration::from_millis(100)).await;
        if logs_contain("TLS handshake failed")
            || logs_contain("invalid peer certificate: UnknownIssuer")
            || logs_contain(
                "Connect error: Custom { kind: InvalidData, error: InvalidCertificate(UnknownIssuer) }",
            )
        {
            failure_detected = true;
            break;
        }
    }

    // Clean up
    let _ = fs::remove_file(&ca_path);

    assert!(
        failure_detected,
        "Should have detected a TLS handshake failure in the logs"
    );
}
