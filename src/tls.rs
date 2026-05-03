// SPDX-License-Identifier: Apache-2.0

use crate::error::{Pv2MqttError, Result};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

pub async fn load_native_certs() -> Result<Arc<rustls::RootCertStore>> {
    tokio::task::spawn_blocking(|| {
        let mut root_cert_store = rustls::RootCertStore::empty();
        let certs = rustls_native_certs::load_native_certs();
        for cert in certs.certs {
            if let Err(e) = root_cert_store.add(cert) {
                tracing::warn!("Could not add a native certificate: {}", e);
            }
        }
        if !certs.errors.is_empty() {
            tracing::warn!(
                "Some native certificates could not be loaded: {:?}",
                certs.errors
            );
        }
        Ok(Arc::new(root_cert_store))
    })
    .await
    .map_err(|e| Pv2MqttError::Internal(format!("Failed to load native certificates: {}", e)))?
}

pub fn create_client_config(
    root_cert_store: Arc<rustls::RootCertStore>,
    ca_path: Option<&str>,
    cert_path: Option<&str>,
    key_path: Option<&str>,
) -> Result<rustls::ClientConfig> {
    let store = if let Some(cp) = ca_path {
        let mut custom_store = rustls::RootCertStore::empty();
        for cert in load_certs(cp)? {
            custom_store.add(cert).map_err(|e| {
                Pv2MqttError::Config(format!("Failed to add CA certificate: {}", e))
            })?;
        }
        custom_store
    } else {
        (*root_cert_store).clone()
    };

    let builder = rustls::ClientConfig::builder().with_root_certificates(store);

    if let (Some(cp), Some(kp)) = (cert_path, key_path) {
        let certs = load_certs(cp)?;
        let key = load_private_key(kp)?;
        builder
            .with_client_auth_cert(certs, key)
            .map_err(|e| Pv2MqttError::Config(format!("Failed to set client auth: {}", e)))
    } else {
        Ok(builder.with_no_client_auth())
    }
}

pub fn load_certs<P: AsRef<Path>>(path: P) -> Result<Vec<CertificateDer<'static>>> {
    let file = File::open(path.as_ref()).map_err(|e| {
        Pv2MqttError::Config(format!(
            "Failed to open certificate file {:?}: {}",
            path.as_ref(),
            e
        ))
    })?;
    let mut reader = BufReader::new(file);
    rustls_pemfile::certs(&mut reader)
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| Pv2MqttError::Config(format!("Failed to parse certificates: {}", e)))
}

pub fn load_private_key<P: AsRef<Path>>(path: P) -> Result<PrivateKeyDer<'static>> {
    let file = File::open(path.as_ref()).map_err(|e| {
        Pv2MqttError::Config(format!(
            "Failed to open private key file {:?}: {}",
            path.as_ref(),
            e
        ))
    })?;
    let mut reader = BufReader::new(file);
    rustls_pemfile::private_key(&mut reader)
        .map_err(|e| Pv2MqttError::Config(format!("Failed to parse private key: {}", e)))?
        .ok_or_else(|| Pv2MqttError::Config("No private key found in file".to_string()))
}
