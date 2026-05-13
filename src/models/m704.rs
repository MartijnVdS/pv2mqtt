// SPDX-License-Identifier: Apache-2.0

use super::traits::SunSpecModel;
use super::types::{ControlData, InverterData, apply_sf};
use sunspec::models::model704::Model704;

impl SunSpecModel for Model704 {
    fn into_inverter_data(self, data: &mut InverterData) {
        use sunspec::models::model704::WMaxLimPctEna;
        data.controls = Some(ControlData {
            conn: None, // Model 704 doesn't have a simple Conn connect/disconnect like 123
            w_max_lim_pct: apply_sf(self.w_max_lim_pct, self.w_max_lim_pct_sf),
            w_max_lim_ena: match self.w_max_lim_pct_ena {
                Some(WMaxLimPctEna::Disabled) => Some(false),
                Some(WMaxLimPctEna::Enabled) => Some(true),
                _ => None,
            },
        });
    }
}
