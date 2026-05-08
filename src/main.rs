// SPDX-License-Identifier: Apache-2.0

#[tokio::main(flavor = "current_thread")]
async fn main() -> pv2mqtt::error::Result<()> {
    pv2mqtt::run().await
}
