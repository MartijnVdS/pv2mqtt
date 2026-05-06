// SPDX-License-Identifier: Apache-2.0

use super::traits::{SunSpecModel, ToStatusString};
use super::types::InverterData;
use sunspec::models::model113::Model113;

impl_to_status_string_for_st!(
    sunspec::models::model113::St,
    Off,
    Sleeping,
    Starting,
    Mppt,
    Throttled,
    ShuttingDown,
    Fault,
    Standby
);

impl SunSpecModel for Model113 {
    fn into_inverter_data(self, data: &mut InverterData) {
        data.aph_a = Some(self.aph_a);
        data.aph_b = Some(self.aph_b);
        data.aph_c = Some(self.aph_c);
        data.v_an = Some(self.ph_vph_a);
        data.v_bn = Some(self.ph_vph_b);
        data.v_cn = Some(self.ph_vph_c);
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
    use sunspec::models::model113::{
        Evt1 as Evt1_113, Evt2 as Evt2_113, EvtVnd1 as EvtVnd1_113, EvtVnd2 as EvtVnd2_113,
        EvtVnd3 as EvtVnd3_113, EvtVnd4 as EvtVnd4_113, St as St113,
    };

    fn empty_m113() -> Model113 {
        Model113 {
            a: 0.0,
            aph_a: 0.0,
            aph_b: 0.0,
            aph_c: 0.0,
            pp_vph_ab: Some(0.0),
            pp_vph_bc: Some(0.0),
            pp_vph_ca: Some(0.0),
            ph_vph_a: 0.0,
            ph_vph_b: 0.0,
            ph_vph_c: 0.0,
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
            st: St113::Off,
            st_vnd: Some(0),
            evt1: Evt1_113::empty(),
            evt2: Evt2_113::empty(),
            evt_vnd1: Some(EvtVnd1_113::empty()),
            evt_vnd2: Some(EvtVnd2_113::empty()),
            evt_vnd3: Some(EvtVnd3_113::empty()),
            evt_vnd4: Some(EvtVnd4_113::empty()),
            hz: 0.0,
        }
    }

    #[test]
    fn test_model113_conversion() {
        let mut m113 = empty_m113();
        m113.a = 30.0;
        m113.w = 12000.0;
        m113.st = St113::Mppt;

        let mut data = InverterData::default();
        m113.into_inverter_data(&mut data);
        assert_relative_eq!(data.w, 12000.0);
        assert_eq!(data.st, "MPPT");
    }
}
