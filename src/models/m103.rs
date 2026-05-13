// SPDX-License-Identifier: Apache-2.0

use super::traits::{SunSpecModel, ToStatusString};
use super::types::{InverterData, apply_sf, apply_sf_f64};
use crate::impl_to_status_string_for_st;
use sunspec::models::model103::Model103;

impl_to_status_string_for_st!(
    sunspec::models::model103::St,
    Off,
    Sleeping,
    Starting,
    Mppt,
    Throttled,
    ShuttingDown,
    Fault,
    Standby
);

impl SunSpecModel for Model103 {
    fn into_inverter_data(self, data: &mut InverterData) {
        data.aph_a = apply_sf(self.aph_a, self.a_sf);
        data.aph_b = apply_sf(self.aph_b, self.a_sf);
        data.aph_c = apply_sf(self.aph_c, self.a_sf);
        data.v_an = apply_sf(self.ph_vph_a, self.v_sf);
        data.v_bn = apply_sf(self.ph_vph_b, self.v_sf);
        data.v_cn = apply_sf(self.ph_vph_c, self.v_sf);
        data.w = apply_sf(self.w, self.w_sf).unwrap_or(0.0);
        data.va = apply_sf(self.va, self.va_sf);
        data.v_ar = apply_sf(self.v_ar, self.v_ar_sf);
        data.wh = apply_sf_f64(self.wh, self.wh_sf).unwrap_or(0.0);
        data.pf = apply_sf(self.pf, self.pf_sf);
        data.hz = apply_sf(self.hz, self.hz_sf).unwrap_or(0.0);
        data.st = self.st.to_status_string();
        data.tmp_cab = apply_sf(self.tmp_cab, self.tmp_sf);
        data.tmp_snk = apply_sf(self.tmp_snk, self.tmp_sf);
        data.tmp_trns = apply_sf(self.tmp_trns, self.tmp_sf);
        data.tmp_ot = apply_sf(self.tmp_ot, self.tmp_sf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sunspec::models::model103::{
        Evt1 as Evt1_103, Evt2 as Evt2_103, EvtVnd1 as EvtVnd1_103, EvtVnd2 as EvtVnd2_103,
        EvtVnd3 as EvtVnd3_103, EvtVnd4 as EvtVnd4_103, St as St103,
    };

    fn empty_m103() -> Model103 {
        Model103 {
            a: 0,
            aph_a: 0,
            aph_b: 0,
            aph_c: 0,
            a_sf: 0,
            pp_vph_ab: None,
            pp_vph_bc: None,
            pp_vph_ca: None,
            ph_vph_a: 0,
            ph_vph_b: 0,
            ph_vph_c: 0,
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
            st: St103::Off,
            st_vnd: Some(0),
            evt1: Evt1_103::empty(),
            evt2: Evt2_103::empty(),
            evt_vnd1: Some(EvtVnd1_103::empty()),
            evt_vnd2: Some(EvtVnd2_103::empty()),
            evt_vnd3: Some(EvtVnd3_103::empty()),
            evt_vnd4: Some(EvtVnd4_103::empty()),
            hz: 0,
            hz_sf: 0,
        }
    }

    #[test]
    fn test_model103_conversion() {
        let mut m103 = empty_m103();
        m103.a = 300;
        m103.a_sf = -1;
        m103.ph_vph_a = 2300;
        m103.v_sf = -1;
        m103.w = 120;
        m103.w_sf = 2;
        m103.st = St103::Mppt;

        let mut data = InverterData::default();
        m103.into_inverter_data(&mut data);
        assert_relative_eq!(data.v_an.unwrap(), 230.0);
        assert_relative_eq!(data.w, 12000.0);
        assert_eq!(data.st, "MPPT");
    }
}
