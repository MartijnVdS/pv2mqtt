// SPDX-License-Identifier: Apache-2.0

use super::traits::{SunSpecModel, ToStatusString};
use super::types::{
    InverterData, apply_sf, apply_sf_i16, apply_sf_i16_opt, apply_sf_opt, apply_sf_u32_f64,
};
use sunspec::models::model102::Model102;

impl ToStatusString for sunspec::models::model102::St {
    fn to_status_string(&self) -> String {
        match self {
            Self::Off => "OFF",
            Self::Sleeping => "SLEEPING",
            Self::Starting => "STARTING",
            Self::Mppt => "MPPT",
            Self::Throttled => "THROTTLED",
            Self::ShuttingDown => "SHUTTING_DOWN",
            Self::Fault => "FAULT",
            Self::Standby => "STANDBY",
            _ => "UNKNOWN",
        }
        .to_string()
    }
}

impl SunSpecModel for Model102 {
    fn into_inverter_data(self, data: &mut InverterData) {
        data.aph_a = apply_sf(self.aph_a, self.a_sf);
        data.aph_b = apply_sf_opt(Some(self.aph_b), self.a_sf);
        data.aph_c = apply_sf_opt(self.aph_c, self.a_sf);
        data.v_an = apply_sf(self.ph_vph_a, self.v_sf);
        data.v_bn = apply_sf_opt(Some(self.ph_vph_b), self.v_sf);
        data.v_cn = apply_sf_opt(self.ph_vph_c, self.v_sf);
        data.w = apply_sf_i16(self.w, self.w_sf).unwrap_or(0.0);
        data.va = apply_sf_i16_opt(self.va, self.va_sf);
        data.v_ar = apply_sf_i16_opt(self.v_ar, self.v_ar_sf);
        data.wh = apply_sf_u32_f64(self.wh, self.wh_sf).unwrap_or(0.0);
        data.pf = apply_sf_i16_opt(self.pf, self.pf_sf);
        data.hz = apply_sf(self.hz, self.hz_sf).unwrap_or(0.0);
        data.st = self.st.to_status_string();
        data.tmp_cab = apply_sf_i16(self.tmp_cab, self.tmp_sf);
        data.tmp_snk = apply_sf_i16_opt(self.tmp_snk, Some(self.tmp_sf));
        data.tmp_trns = apply_sf_i16_opt(self.tmp_trns, Some(self.tmp_sf));
        data.tmp_ot = apply_sf_i16_opt(self.tmp_ot, Some(self.tmp_sf));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sunspec::models::model102::{
        Evt1 as Evt1_102, Evt2 as Evt2_102, EvtVnd1 as EvtVnd1_102, EvtVnd2 as EvtVnd2_102,
        EvtVnd3 as EvtVnd3_102, EvtVnd4 as EvtVnd4_102, St as St102,
    };

    fn empty_m102() -> Model102 {
        Model102 {
            a: 0,
            aph_a: 0,
            aph_b: 0,
            aph_c: None,
            a_sf: 0,
            pp_vph_ab: None,
            pp_vph_bc: None,
            pp_vph_ca: None,
            ph_vph_a: 0,
            ph_vph_b: 0,
            ph_vph_c: None,
            v_sf: 0,
            w: 0,
            w_sf: 0,
            va: Some(0),
            va_sf: Some(0),
            v_ar: Some(0),
            v_ar_sf: Some(0),
            pf: Some(0),
            pf_sf: Some(0),
            wh: 0,
            wh_sf: 0,
            dca: Some(0),
            dca_sf: Some(0),
            dcv: Some(0),
            dcv_sf: Some(0),
            dcw: Some(0),
            dcw_sf: Some(0),
            tmp_cab: 0,
            tmp_snk: Some(0),
            tmp_trns: Some(0),
            tmp_ot: Some(0),
            tmp_sf: 0,
            st: St102::Off,
            st_vnd: Some(0),
            evt1: Evt1_102::empty(),
            evt2: Evt2_102::empty(),
            evt_vnd1: Some(EvtVnd1_102::empty()),
            evt_vnd2: Some(EvtVnd2_102::empty()),
            evt_vnd3: Some(EvtVnd3_102::empty()),
            evt_vnd4: Some(EvtVnd4_102::empty()),
            hz: 0,
            hz_sf: 0,
        }
    }

    #[test]
    fn test_model102_conversion() {
        let mut m102 = empty_m102();
        m102.a = 500;
        m102.a_sf = -1;
        m102.ph_vph_a = 2300;
        m102.v_sf = -1;
        m102.w = 100;
        m102.w_sf = 2;
        m102.st = St102::Mppt;

        let mut data = InverterData::default();
        m102.into_inverter_data(&mut data);
        assert_relative_eq!(data.v_an.unwrap(), 230.0);
        assert_relative_eq!(data.w, 10000.0);
        assert_eq!(data.st, "MPPT");
    }
}
