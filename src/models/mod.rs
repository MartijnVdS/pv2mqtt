// SPDX-License-Identifier: Apache-2.0

pub mod m101;
pub mod m102;
pub mod m103;
pub mod m111;
pub mod m112;
pub mod m113;
pub mod m123;
pub mod traits;
pub mod types;

pub use traits::*;
pub use types::*;

use crate::error::{Pv2MqttError, Result};
use std::sync::Arc;
use sunspec::client::AsyncDevice;
use tokio::sync::Mutex;
use tokio_modbus::client::Context as ModbusContext;

pub static SUPPORTED_MODELS: &[u16] = &[101, 102, 103, 111, 112, 113, 123];

/// Reads a specific SunSpec model and applies its data to the provided `InverterData` struct.
/// Uses a mutable reference because SunSpec devices often represent a single inverter
/// across multiple models (e.g., electrical data in 103, controls in 123), and this allows
/// us to additively compose the final data structure without complex merging or reallocations.
pub async fn poll_and_apply(
    model_id: u16,
    device: &AsyncDevice<Arc<Mutex<ModbusContext>>>,
    data: &mut InverterData,
) -> Result<()> {
    match model_id {
        101 => {
            let m = device
                .read_model::<sunspec::models::model101::Model101>()
                .await?;
            m.into_inverter_data(data);
        }
        102 => {
            let m = device
                .read_model::<sunspec::models::model102::Model102>()
                .await?;
            m.into_inverter_data(data);
        }
        103 => {
            let m = device
                .read_model::<sunspec::models::model103::Model103>()
                .await?;
            m.into_inverter_data(data);
        }
        111 => {
            let m = device
                .read_model::<sunspec::models::model111::Model111>()
                .await?;
            m.into_inverter_data(data);
        }
        112 => {
            let m = device
                .read_model::<sunspec::models::model112::Model112>()
                .await?;
            m.into_inverter_data(data);
        }
        113 => {
            let m = device
                .read_model::<sunspec::models::model113::Model113>()
                .await?;
            m.into_inverter_data(data);
        }
        123 => {
            let m = device
                .read_model::<sunspec::models::model123::Model123>()
                .await?;
            m.into_inverter_data(data);
        }
        _ => return Err(Pv2MqttError::UnsupportedModel(model_id)),
    }
    Ok(())
}
