// SPDX-License-Identifier: Apache-2.0

use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main(flavor = "current_thread")]
async fn main() -> pv2mqtt::error::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    // Install default crypto provider for rustls
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .ok();

    info!("Starting pv2mqtt");

    pv2mqtt::run().await
}
