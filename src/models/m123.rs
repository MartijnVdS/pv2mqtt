// SPDX-License-Identifier: Apache-2.0

use super::traits::SunSpecModel;
use super::types::{ControlData, InverterData, apply_sf};
use sunspec::models::model123::Model123;

impl SunSpecModel for Model123 {
    fn into_inverter_data(self, data: &mut InverterData) {
        use sunspec::models::model123::{Conn, WMaxLimEna};
        data.controls = Some(ControlData {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::traits::SunSpecModel;
    use crate::models::types::InverterData;
    use approx::assert_relative_eq;
    use sunspec::models::model123::{Conn, WMaxLimEna};

    #[test]
    fn test_model123_conversion() {
        let m123 = Model123 {
            conn: Conn::Connect,
            w_max_lim_pct: 7550,
            w_max_lim_pct_sf: -2,
            w_max_lim_ena: WMaxLimEna::Enabled,
            ..empty_m123()
        };

        let mut data = InverterData::default();
        m123.into_inverter_data(&mut data);

        let controls = data.controls.unwrap();
        assert_eq!(controls.conn, Some(true));
        assert_relative_eq!(controls.w_max_lim_pct.unwrap(), 75.5);
        assert_eq!(controls.w_max_lim_ena, Some(true));
    }

    #[test]
    fn test_model123_conversion_disabled() {
        let m123 = Model123 {
            conn: Conn::Disconnect,
            w_max_lim_pct: 100,
            w_max_lim_pct_sf: 0,
            w_max_lim_ena: WMaxLimEna::Disabled,
            ..empty_m123()
        };

        let mut data = InverterData::default();
        m123.into_inverter_data(&mut data);

        let controls = data.controls.unwrap();
        assert_eq!(controls.conn, Some(false));
        assert_relative_eq!(controls.w_max_lim_pct.unwrap(), 100.0);
        assert_eq!(controls.w_max_lim_ena, Some(false));
    }

    fn empty_m123() -> Model123 {
        use sunspec::models::model123::{Conn, OutPfSetEna, VArPctEna, WMaxLimEna};
        Model123 {
            conn_win_tms: None,
            conn_rvrt_tms: None,
            conn: Conn::Connect,
            w_max_lim_pct: 0,
            w_max_lim_pct_win_tms: None,
            w_max_lim_pct_rvrt_tms: None,
            w_max_lim_pct_rmp_tms: None,
            w_max_lim_ena: WMaxLimEna::Disabled,
            out_pf_set: 0,
            out_pf_set_win_tms: None,
            out_pf_set_rvrt_tms: None,
            out_pf_set_rmp_tms: None,
            out_pf_set_ena: OutPfSetEna::Disabled,
            v_ar_w_max_pct: None,
            v_ar_max_pct: None,
            v_ar_aval_pct: None,
            v_ar_pct_win_tms: None,
            v_ar_pct_rvrt_tms: None,
            v_ar_pct_rmp_tms: None,
            v_ar_pct_mod: None,
            v_ar_pct_ena: VArPctEna::Disabled,
            w_max_lim_pct_sf: 0,
            out_pf_set_sf: 0,
            v_ar_pct_sf: None,
        }
    }
}
