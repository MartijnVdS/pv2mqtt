// SPDX-License-Identifier: Apache-2.0

use super::traits::{SunSpecModel, ToStatusString};
use super::types::InverterData;
use crate::impl_to_status_string_for_st;
use sunspec::models::model112::Model112;

impl_to_status_string_for_st!(
    sunspec::models::model112::St,
    Off,
    Sleeping,
    Starting,
    Mppt,
    Throttled,
    ShuttingDown,
    Fault,
    Standby
);

impl SunSpecModel for Model112 {
    fn into_inverter_data(self, data: &mut InverterData) {
        data.aph_a = Some(self.aph_a);
        data.aph_b = Some(self.aph_b);
        data.aph_c = self.aph_c;
        data.v_an = Some(self.ph_vph_a);
        data.v_bn = Some(self.ph_vph_b);
        data.v_cn = self.ph_vph_c;
        data.w = self.w;
        data.va = self.va;
        data.v_ar = self.v_ar;
        data.wh = self.wh as f64;
        data.pf = self.pf;
        data.hz = self.hz;
        data.st = self.st.to_status_string();
        data.tmp_cab = Some(self.tmp_cab);
        data.tmp_snk = self.tmp_snk;
        data.tmp_trns = self.tmp_trns;
        data.tmp_ot = self.tmp_ot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sunspec::models::model112::{
        Evt1 as Evt1_112, Evt2 as Evt2_112, EvtVnd1 as EvtVnd1_112, EvtVnd2 as EvtVnd2_112,
        EvtVnd3 as EvtVnd3_112, EvtVnd4 as EvtVnd4_112, St as St112,
    };

    fn empty_m112() -> Model112 {
        Model112 {
            a: 0.0,
            aph_a: 0.0,
            aph_b: 0.0,
            aph_c: None,
            pp_vph_ab: Some(0.0),
            pp_vph_bc: None,
            pp_vph_ca: None,
            ph_vph_a: 0.0,
            ph_vph_b: 0.0,
            ph_vph_c: None,
            w: 0.0,
            va: Some(0.0),
            v_ar: Some(0.0),
            pf: Some(0.0),
            wh: 0.0,
            dca: Some(0.0),
            dcv: Some(0.0),
            dcw: Some(0.0),
            tmp_cab: 0.0,
            tmp_snk: Some(0.0),
            tmp_trns: Some(0.0),
            tmp_ot: Some(0.0),
            st: St112::Off,
            st_vnd: Some(0),
            evt1: Evt1_112::empty(),
            evt2: Evt2_112::empty(),
            evt_vnd1: Some(EvtVnd1_112::empty()),
            evt_vnd2: Some(EvtVnd2_112::empty()),
            evt_vnd3: Some(EvtVnd3_112::empty()),
            evt_vnd4: Some(EvtVnd4_112::empty()),
            hz: 0.0,
        }
    }

    #[test]
    fn test_model112_conversion() {
        let mut m112 = empty_m112();
        m112.a = 20.0;
        m112.w = 8000.0;
        m112.st = St112::Mppt;

        let mut data = InverterData::default();
        m112.into_inverter_data(&mut data);
        assert_relative_eq!(data.w, 8000.0);
        assert_eq!(data.st, "MPPT");
    }
}
