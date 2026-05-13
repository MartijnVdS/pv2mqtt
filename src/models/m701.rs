// SPDX-License-Identifier: Apache-2.0

use super::traits::{SunSpecModel, ToStatusString};
use super::types::{InverterData, SUNSPEC_UNIMPLEMENTED_U16, apply_sf, apply_sf_f64};
use sunspec::models::model701::{
    Alrm as Alrm701, DerMode as DerMode701, Model701, ThrotSrc as ThrotSrc701,
};

impl ToStatusString for sunspec::models::model701::InvSt {
    fn to_status_string(&self) -> String {
        match self {
            Self::Off => "OFF",
            Self::Sleeping => "SLEEPING",
            Self::Starting => "STARTING",
            Self::Running => "RUNNING",
            Self::Throttled => "THROTTLED",
            Self::ShuttingDown => "SHUTTING_DOWN",
            Self::Fault => "FAULT",
            Self::Standby => "STANDBY",
            _ => "UNKNOWN",
        }
        .to_string()
    }
}

impl ToStatusString for sunspec::models::model701::ConnSt {
    fn to_status_string(&self) -> String {
        match self {
            Self::Disconnected => "DISCONNECTED",
            Self::Connected => "CONNECTED",
            _ => "UNKNOWN",
        }
        .to_string()
    }
}

fn map_alrm(alrm: Alrm701) -> Option<Vec<String>> {
    crate::map_sunspec_flags!(
        alrm,
        Alrm701,
        [
            GroundFault => "GROUND_FAULT",
            DcOverVolt => "DC_OVER_VOLT",
            AcDisconnect => "AC_DISCONNECT",
            DcDisconnect => "DC_DISCONNECT",
            GridDisconnect => "GRID_DISCONNECT",
            CabinetOpen => "CABINET_OPEN",
            ManualShutdown => "MANUAL_SHUTDOWN",
            OverTemp => "OVER_TEMP",
            OverFrequency => "OVER_FREQUENCY",
            UnderFrequency => "UNDER_FREQUENCY",
            AcOverVolt => "AC_OVER_VOLT",
            AcUnderVolt => "AC_UNDER_VOLT",
            BlownStringFuse => "BLOWN_STRING_FUSE",
            UnderTemp => "UNDER_TEMP",
            MemoryLoss => "MEMORY_LOSS",
            HwTestFailure => "HW_TEST_FAILURE",
            ManufacturerAlrm => "MANUFACTURER_ALRM",
        ]
    )
}

fn map_der_mode(mode: DerMode701) -> Option<Vec<String>> {
    crate::map_sunspec_flags!(
        mode,
        DerMode701,
        [
            GridFollowing => "GRID_FOLLOWING",
            GridForming => "GRID_FORMING",
            PvClipped => "PV_CLIPPED",
        ]
    )
}

fn map_throt_src(src: ThrotSrc701) -> Option<Vec<String>> {
    crate::map_sunspec_flags!(
        src,
        ThrotSrc701,
        [
            MaxW => "MAX_W",
            FixedW => "FIXED_W",
            FixedVar => "FIXED_VAR",
            FixedPf => "FIXED_PF",
            VoltVar => "VOLT_VAR",
            FreqWatt => "FREQ_WATT",
            DynReactCurr => "DYN_REACT_CURR",
            Lvrt => "LVRT",
            Hvrt => "HVRT",
            WattVar => "WATT_VAR",
            VoltWatt => "VOLT_WATT",
            Scheduled => "SCHEDULED",
            Lfrt => "LFRT",
            Hfrt => "HFRT",
            Derated => "DERATED",
        ]
    )
}

impl SunSpecModel for Model701 {
    fn into_inverter_data(self, data: &mut InverterData) {
        data.aph_a = apply_sf(self.al1, self.a_sf);
        data.aph_b = apply_sf(self.al2, self.a_sf);
        data.aph_c = apply_sf(self.al3, self.a_sf);
        data.v_an = apply_sf(self.vl1, self.v_sf);
        data.v_bn = apply_sf(self.vl2, self.v_sf);
        data.v_cn = apply_sf(self.vl3, self.v_sf);
        data.w = apply_sf(self.w, self.w_sf).unwrap_or(0.0);
        data.va = apply_sf(self.va, self.va_sf);
        data.v_ar = apply_sf(self.var, self.var_sf);
        data.wh = apply_sf_f64(self.tot_wh_inj, self.tot_wh_sf).unwrap_or(0.0);
        data.pf = apply_sf(self.pf, self.pf_sf);
        data.hz = apply_sf(self.hz, self.hz_sf).unwrap_or(0.0);
        data.st = self
            .inv_st
            .as_ref()
            .map(|s| s.to_status_string())
            .unwrap_or_else(|| "UNKNOWN".to_string());
        data.tmp_cab = apply_sf(self.tmp_cab, self.tmp_sf);
        data.tmp_snk = apply_sf(self.tmp_snk, self.tmp_sf);
        data.tmp_trns = apply_sf(self.tmp_trns, self.tmp_sf);
        data.tmp_ot = apply_sf(self.tmp_ot, self.tmp_sf);
        data.tmp_amb = apply_sf(self.tmp_amb, self.tmp_sf);
        data.tmp_sw = apply_sf(self.tmp_sw, self.tmp_sf);

        data.alrm = self.alrm.and_then(map_alrm);
        data.der_mode = self.der_mode.and_then(map_der_mode);
        data.conn_st = self.conn_st.and_then(|c| match c {
            sunspec::models::model701::ConnSt::Connected => Some(true),
            sunspec::models::model701::ConnSt::Disconnected => Some(false),
            _ => None,
        });

        data.throt_pct = self.throt_pct.and_then(|p| {
            if p == SUNSPEC_UNIMPLEMENTED_U16 {
                None
            } else {
                Some(p as f32)
            }
        });
        data.throt_src = self.throt_src.and_then(map_throt_src);

        data.mn_alrm_info = self.mn_alrm_info.and_then(|s| {
            let cleaned = s.trim_matches(char::from(0)).trim().to_string();
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sunspec::models::model701::{
        Alrm as Alrm701, DerMode as DerMode701, InvSt as InvSt701, ThrotSrc as ThrotSrc701,
    };

    fn empty_m701() -> Model701 {
        Model701 {
            ac_type: sunspec::models::model701::AcType::ThreePhase,
            st: Some(sunspec::models::model701::St::On),
            inv_st: Some(InvSt701::Running),
            conn_st: Some(sunspec::models::model701::ConnSt::Connected),
            alrm: Some(Alrm701::empty()),
            der_mode: Some(DerMode701::empty()),
            w: Some(0),
            va: Some(0),
            var: Some(0),
            pf: Some(0),
            a: Some(0),
            llv: Some(0),
            lnv: Some(0),
            hz: Some(0),
            tot_wh_inj: Some(0),
            tot_wh_abs: Some(0),
            tot_varh_inj: Some(0),
            tot_varh_abs: Some(0),
            tmp_amb: Some(0),
            tmp_cab: Some(0),
            tmp_snk: Some(0),
            tmp_trns: Some(0),
            tmp_sw: Some(0),
            tmp_ot: Some(0),
            wl1: Some(0),
            val1: Some(0),
            var_l1: Some(0),
            pfl1: Some(0),
            al1: Some(0),
            vl1l2: Some(0),
            vl1: Some(0),
            tot_wh_inj_l1: Some(0),
            tot_wh_abs_l1: Some(0),
            tot_varh_inj_l1: Some(0),
            tot_varh_abs_l1: Some(0),
            wl2: Some(0),
            val2: Some(0),
            var_l2: Some(0),
            pfl2: Some(0),
            al2: Some(0),
            vl2l3: Some(0),
            vl2: Some(0),
            tot_wh_inj_l2: Some(0),
            tot_wh_abs_l2: Some(0),
            tot_varh_inj_l2: Some(0),
            tot_varh_abs_l2: Some(0),
            wl3: Some(0),
            val3: Some(0),
            var_l3: Some(0),
            pfl3: Some(0),
            al3: Some(0),
            vl3l1: Some(0),
            vl3: Some(0),
            tot_wh_inj_l3: Some(0),
            tot_wh_abs_l3: Some(0),
            tot_varh_inj_l3: Some(0),
            tot_varh_abs_l3: Some(0),
            throt_pct: Some(0),
            throt_src: Some(ThrotSrc701::empty()),
            a_sf: Some(0),
            v_sf: Some(0),
            hz_sf: Some(0),
            w_sf: Some(0),
            pf_sf: Some(0),
            va_sf: Some(0),
            var_sf: Some(0),
            tot_wh_sf: Some(0),
            tot_varh_sf: Some(0),
            tmp_sf: Some(0),
            mn_alrm_info: Some(String::new()),
        }
    }

    #[test]
    fn test_model701_conversion() {
        let mut m701 = empty_m701();
        m701.al1 = Some(100);
        m701.a_sf = Some(-1);
        m701.vl1 = Some(2300);
        m701.v_sf = Some(-1);
        m701.w = Some(500);
        m701.w_sf = Some(1);
        m701.inv_st = Some(InvSt701::Running);
        m701.alrm = Some(Alrm701::GroundFault | Alrm701::DcOverVolt);
        m701.conn_st = Some(sunspec::models::model701::ConnSt::Connected);
        m701.mn_alrm_info = Some("Test Alarm\0\0".to_string());
        m701.throt_pct = Some(50);

        let mut data = InverterData::default();
        m701.into_inverter_data(&mut data);
        assert_relative_eq!(data.aph_a.unwrap(), 10.0);
        assert_relative_eq!(data.v_an.unwrap(), 230.0);
        assert_relative_eq!(data.w, 5000.0);
        assert_eq!(data.st, "RUNNING");
        assert_eq!(data.conn_st, Some(true));
        assert_eq!(data.mn_alrm_info, Some("Test Alarm".to_string()));
        assert_relative_eq!(data.throt_pct.unwrap(), 50.0);
        let alrms = data.alrm.unwrap();
        assert!(alrms.contains(&"GROUND_FAULT".to_string()));
        assert!(alrms.contains(&"DC_OVER_VOLT".to_string()));
    }
}
