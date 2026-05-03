// SPDX-License-Identifier: Apache-2.0

use super::traits::SunSpecModel;
use super::types::{InverterData, Model123Data, apply_sf};
use sunspec::models::model123::Model123;

impl SunSpecModel for Model123 {
    fn into_inverter_data(self, data: &mut InverterData) {
        use sunspec::models::model123::{Conn, WMaxLimEna};
        data.controls = Some(Model123Data {
            conn: match self.conn {
                Conn::Disconnect => Some(false),
                Conn::Connect => Some(true),
                _ => None,
            },
            w_max_lim_pct: apply_sf(self.w_max_lim_pct, self.w_max_lim_pct_sf),
            w_max_lim_ena: match self.w_max_lim_ena {
                WMaxLimEna::Disabled => Some(false),
                WMaxLimEna::Enabled => Some(true),
                _ => None,
            },
        });
    }
}
