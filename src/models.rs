// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Utc};
use serde::Serialize;
use sunspec::models::{
    model101::Model101, model102::Model102, model103::Model103, model111::Model111,
    model112::Model112, model113::Model113,
};

pub static SUPPORTED_MODELS: &[u16] = &[101, 102, 103, 111, 112, 113];

const SUNSPEC_UNIMPLEMENTED_U32 : u32 = 0xFFFFFFFF;
const SUNSPEC_UNIMPLEMENTED_U16 : u16 = 0xFFFF;
const SUNSPEC_UNIMPLEMENTED_I16 : i16 = -32768;

#[derive(Debug, Serialize, Default, Clone)]
pub struct InverterData {
    pub timestamp: DateTime<Utc>,
    #[serde(rename = "AphA")]
    pub aph_a: Option<f32>,
    #[serde(rename = "AphB")]
    pub aph_b: Option<f32>,
    #[serde(rename = "AphC")]
    pub aph_c: Option<f32>,
    #[serde(rename = "PhVphA")]
    pub v_an: Option<f32>,
    #[serde(rename = "PhVphB")]
    pub v_bn: Option<f32>,
    #[serde(rename = "PhVphC")]
    pub v_cn: Option<f32>,
    #[serde(rename = "W")]
    pub w: f32,
    #[serde(rename = "VA")]
    pub va: Option<f32>,
    #[serde(rename = "VAr")]
    pub v_ar: Option<f32>,
    #[serde(rename = "WH")]
    pub wh: f32,
    #[serde(rename = "PF")]
    pub pf: Option<f32>,
    #[serde(rename = "Hz")]
    pub hz: f32,
    #[serde(rename = "TmpCab")]
    pub tmp_cab: Option<f32>,
    #[serde(rename = "TmpSnk")]
    pub tmp_snk: Option<f32>,
    #[serde(rename = "TmpTrns")]
    pub tmp_trns: Option<f32>,
    #[serde(rename = "TmpOt")]
    pub tmp_ot: Option<f32>,
    #[serde(rename = "St")]
    pub st: String,
}

fn apply_sf(val: u16, sf: i16) -> Option<f32> {
    if val == SUNSPEC_UNIMPLEMENTED_U16 {
        return None;
    }
    Some(val as f32 * 10f32.powi(sf as i32))
}

fn apply_sf_i16(val: i16, sf: i16) -> Option<f32> {
    if val == SUNSPEC_UNIMPLEMENTED_I16 {
        return None;
    }
    Some(val as f32 * 10f32.powi(sf as i32))
}

fn apply_sf_u32(val: u32, sf: i16) -> Option<f32> {
    if val == SUNSPEC_UNIMPLEMENTED_U32 {
        return None;
    }
    Some(val as f32 * 10f32.powi(sf as i32))
}

fn apply_sf_i16_opt(val: Option<i16>, sf: Option<i16>) -> Option<f32> {
    match (val, sf) {
        (Some(v), Some(s)) => apply_sf_i16(v, s),
        _ => None,
    }
}

fn apply_sf_opt(val: Option<u16>, sf: i16) -> Option<f32> {
    val.and_then(|v| apply_sf(v, sf))
}

trait ToStatusString {
    fn to_status_string(&self) -> String;
}

macro_rules! impl_to_status_string {
    ($($st_path:path),+; $mapping:tt) => {
        $(
            impl_to_status_string_single!($st_path, $mapping);
        )+
    };
}

macro_rules! impl_to_status_string_single {
    ($st_path:path, { $($variant:ident => $val:expr),* $(,)? }) => {
        impl ToStatusString for $st_path {
            fn to_status_string(&self) -> String {
                match self {
                    $( <$st_path>::$variant => $val, )*
                    _ => "UNKNOWN",
                }
                .to_string()
            }
        }
    };
}

impl_to_status_string!(
    sunspec::models::model101::St,
    sunspec::models::model102::St,
    sunspec::models::model103::St,
    sunspec::models::model112::St,
    sunspec::models::model113::St;
    {
        Off => "OFF",
        Sleeping => "SLEEPING",
        Starting => "STARTING",
        Mppt => "MPPT",
        Throttled => "THROTTLED",
        ShuttingDown => "SHUTTING_DOWN",
        Fault => "FAULT",
        Standby => "STANDBY",
    }
);

// For some unknown reason, the "St" type in model 111 is defined with a
// "Gg" prefix even in the official JSON
impl_to_status_string!(sunspec::models::model111::St; {
    GgOff => "OFF",
    GgSleeping => "SLEEPING",
    GgStarting => "STARTING",
    GgMppt => "MPPT",
    GgThrottled => "THROTTLED",
    GgShuttingDown => "SHUTTING_DOWN",
    GgFault => "FAULT",
    GgStandby => "STANDBY",
});

macro_rules! impl_int_model {
    ($model:ident) => {
        impl From<$model> for InverterData {
            fn from(m: $model) -> Self {
                Self {
                    timestamp: Utc::now(),
                    aph_a: apply_sf(m.aph_a, m.a_sf),
                    aph_b: apply_sf_opt(m.aph_b.into(), m.a_sf),
                    aph_c: apply_sf_opt(m.aph_c.into(), m.a_sf),
                    v_an: apply_sf(m.ph_vph_a, m.v_sf),
                    v_bn: apply_sf_opt(m.ph_vph_b.into(), m.v_sf),
                    v_cn: apply_sf_opt(m.ph_vph_c.into(), m.v_sf),
                    w: apply_sf_i16(m.w, m.w_sf).unwrap_or(0.0),
                    va: apply_sf_i16_opt(m.va, m.va_sf),
                    v_ar: apply_sf_i16_opt(m.v_ar, m.v_ar_sf),
                    wh: apply_sf_u32(m.wh, m.wh_sf).unwrap_or(0.0),
                    pf: apply_sf_i16_opt(m.pf, m.pf_sf),
                    hz: apply_sf(m.hz, m.hz_sf).unwrap_or(0.0),
                    st: m.st.to_status_string(),
                    tmp_cab: apply_sf_i16(m.tmp_cab, m.tmp_sf),
                    tmp_snk: apply_sf_i16_opt(m.tmp_snk, Some(m.tmp_sf)),
                    tmp_trns: apply_sf_i16_opt(m.tmp_trns, Some(m.tmp_sf)),
                    tmp_ot: apply_sf_i16_opt(m.tmp_ot, Some(m.tmp_sf)),
                }
            }
        }
    };
}

macro_rules! impl_float_model {
    ($model:ident) => {
        impl From<$model> for InverterData {
            fn from(m: $model) -> Self {
                Self {
                    timestamp: Utc::now(),
                    aph_a: Some(m.aph_a),
                    aph_b: m.aph_b.into(),
                    aph_c: m.aph_c.into(),
                    v_an: Some(m.ph_vph_a),
                    v_bn: m.ph_vph_b.into(),
                    v_cn: m.ph_vph_c.into(),
                    w: m.w,
                    va: m.va,
                    v_ar: m.v_ar,
                    wh: m.wh,
                    pf: m.pf,
                    hz: m.hz,
                    st: m.st.to_status_string(),
                    tmp_cab: Some(m.tmp_cab),
                    tmp_snk: m.tmp_snk,
                    tmp_trns: m.tmp_trns,
                    tmp_ot: m.tmp_ot,
                }
            }
        }
    };
}

impl_int_model!(Model101);
impl_int_model!(Model102);
impl_int_model!(Model103);

impl_float_model!(Model111);
impl_float_model!(Model112);
impl_float_model!(Model113);

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 0.001,
            "actual: {}, expected: {}",
            actual,
            expected
        );
    }

    fn assert_near_opt(actual: Option<f32>, expected: Option<f32>) {
        match (actual, expected) {
            (Some(a), Some(e)) => assert_near(a, e),
            (None, None) => {}
            _ => panic!("actual: {:?}, expected: {:?}", actual, expected),
        }
    }

    use sunspec::models::model101::{
        Evt1 as Evt1_101, Evt2 as Evt2_101, EvtVnd1 as EvtVnd1_101, EvtVnd2 as EvtVnd2_101,
        EvtVnd3 as EvtVnd3_101, EvtVnd4 as EvtVnd4_101, St as St101,
    };
    use sunspec::models::model102::{
        Evt1 as Evt1_102, Evt2 as Evt2_102, EvtVnd1 as EvtVnd1_102, EvtVnd2 as EvtVnd2_102,
        EvtVnd3 as EvtVnd3_102, EvtVnd4 as EvtVnd4_102, St as St102,
    };
    use sunspec::models::model103::{
        Evt1 as Evt1_103, Evt2 as Evt2_103, EvtVnd1 as EvtVnd1_103, EvtVnd2 as EvtVnd2_103,
        EvtVnd3 as EvtVnd3_103, EvtVnd4 as EvtVnd4_103, St as St103,
    };
    use sunspec::models::model111::{
        Evt1 as Evt1_111, Evt2 as Evt2_111, EvtVnd1 as EvtVnd1_111, EvtVnd2 as EvtVnd2_111,
        EvtVnd3 as EvtVnd3_111, EvtVnd4 as EvtVnd4_111, St as St111,
    };
    use sunspec::models::model112::{
        Evt1 as Evt1_112, Evt2 as Evt2_112, EvtVnd1 as EvtVnd1_112, EvtVnd2 as EvtVnd2_112,
        EvtVnd3 as EvtVnd3_112, EvtVnd4 as EvtVnd4_112, St as St112,
    };
    use sunspec::models::model113::{
        Evt1 as Evt1_113, Evt2 as Evt2_113, EvtVnd1 as EvtVnd1_113, EvtVnd2 as EvtVnd2_113,
        EvtVnd3 as EvtVnd3_113, EvtVnd4 as EvtVnd4_113, St as St113,
    };

    fn empty_m101() -> Model101 {
        Model101 {
            a: 0,
            aph_a: 0,
            aph_b: None,
            aph_c: None,
            a_sf: 0,
            pp_vph_ab: None,
            pp_vph_bc: None,
            pp_vph_ca: None,
            ph_vph_a: 0,
            ph_vph_b: None,
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
            st: St101::Off,
            st_vnd: Some(0),
            evt1: Evt1_101::empty(),
            evt2: Evt2_101::empty(),
            evt_vnd1: Some(EvtVnd1_101::empty()),
            evt_vnd2: Some(EvtVnd2_101::empty()),
            evt_vnd3: Some(EvtVnd3_101::empty()),
            evt_vnd4: Some(EvtVnd4_101::empty()),
            hz: 0,
            hz_sf: 0,
        }
    }

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

    fn empty_m111() -> Model111 {
        Model111 {
            a: 0.0,
            aph_a: 0.0,
            aph_b: None,
            aph_c: None,
            pp_vph_ab: None,
            pp_vph_bc: None,
            pp_vph_ca: None,
            ph_vph_a: 0.0,
            ph_vph_b: None,
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
            st: St111::GgOff,
            st_vnd: Some(0),
            evt1: Evt1_111::empty(),
            evt2: Evt2_111::empty(),
            evt_vnd1: Some(EvtVnd1_111::empty()),
            evt_vnd2: Some(EvtVnd2_111::empty()),
            evt_vnd3: Some(EvtVnd3_111::empty()),
            evt_vnd4: Some(EvtVnd4_111::empty()),
            hz: 0.0,
        }
    }

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
    fn test_apply_sf() {
        assert_near_opt(apply_sf(100, 0), Some(100.0));
        assert_near_opt(apply_sf(100, -1), Some(10.0));
        assert_near_opt(apply_sf(100, 1), Some(1000.0));
        assert_eq!(apply_sf(SUNSPEC_UNIMPLEMENTED_U16, -1), None);
    }

    #[test]
    fn test_apply_sf_i16() {
        assert_near_opt(apply_sf_i16(100, 0), Some(100.0));
        assert_near_opt(apply_sf_i16(-100, -1), Some(-10.0));
        assert_eq!(apply_sf_i16(SUNSPEC_UNIMPLEMENTED_I16, -1), None);
    }

    #[test]
    fn test_apply_sf_i16_opt() {
        assert_near_opt(apply_sf_i16_opt(Some(100), Some(0)), Some(100.0));
        assert_near_opt(apply_sf_i16_opt(Some(100), Some(-1)), Some(10.0));
        assert_eq!(apply_sf_i16_opt(None, Some(-1)), None);
        assert_eq!(apply_sf_i16_opt(Some(100), None), None);
        assert_eq!(apply_sf_i16_opt(None, None), None);
        assert_eq!(apply_sf_i16_opt(Some(SUNSPEC_UNIMPLEMENTED_I16), Some(-1)), None);
    }

    #[test]
    fn test_apply_sf_u32() {
        assert_near_opt(apply_sf_u32(100000, 0), Some(100000.0));
        assert_near(apply_sf_u32(123456, -2).unwrap(), 1234.56);
        assert_eq!(apply_sf_u32(SUNSPEC_UNIMPLEMENTED_U32, -2), None);
    }

    #[test]
    fn test_not_implemented_values() {
        let mut m101 = empty_m101();
        m101.ph_vph_a = SUNSPEC_UNIMPLEMENTED_U16; // Not implemented
        m101.w = SUNSPEC_UNIMPLEMENTED_I16; // Not implemented
        m101.va = Some(SUNSPEC_UNIMPLEMENTED_I16); // Not implemented

        let data = InverterData::from(m101);
        assert_eq!(data.v_an, None);
        assert_near(data.w, 0.0);
        assert_eq!(data.va, None);
    }

    #[test]
    fn test_model101_conversion() {
        let mut m101 = empty_m101();
        m101.a = 1000;
        m101.a_sf = -1;
        m101.ph_vph_a = 2300;
        m101.v_sf = -1;
        m101.w = 2300;
        m101.w_sf = 1;
        m101.hz = 5000;
        m101.hz_sf = -2;
        m101.wh = 123456;
        m101.wh_sf = -2;
        m101.st = St101::Mppt;
        m101.tmp_cab = 450;
        m101.tmp_sf = -1;

        let data = InverterData::from(m101);
        assert_near_opt(data.v_an, Some(230.0));
        assert_near(data.w, 23000.0);
        assert_near(data.hz, 50.0);
        assert_near(data.wh, 1234.56);
        assert_eq!(data.st, "MPPT");
        assert_near_opt(data.tmp_cab, Some(45.0));
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

        let data = InverterData::from(m102);
        assert_near_opt(data.v_an, Some(230.0));
        assert_near(data.w, 10000.0);
        assert_eq!(data.st, "MPPT");
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

        let data = InverterData::from(m103);
        assert_near_opt(data.v_an, Some(230.0));
        assert_near(data.w, 12000.0);
        assert_eq!(data.st, "MPPT");
    }

    #[test]
    fn test_model111_conversion() {
        let mut m111 = empty_m111();
        m111.a = 10.5;
        m111.w = 2300.0;
        m111.hz = 50.0;
        m111.st = St111::GgMppt;

        let data = InverterData::from(m111);
        assert_near(data.w, 2300.0);
        assert_near(data.hz, 50.0);
        assert_eq!(data.st, "MPPT");
    }

    #[test]
    fn test_model112_conversion() {
        let mut m112 = empty_m112();
        m112.a = 20.0;
        m112.w = 8000.0;
        m112.st = St112::Mppt;

        let data = InverterData::from(m112);
        assert_near(data.w, 8000.0);
        assert_eq!(data.st, "MPPT");
    }

    #[test]
    fn test_model113_conversion() {
        let mut m113 = empty_m113();
        m113.a = 30.0;
        m113.w = 12000.0;
        m113.st = St113::Mppt;

        let data = InverterData::from(m113);
        assert_near(data.w, 12000.0);
        assert_eq!(data.st, "MPPT");
    }

    #[test]
    fn test_serialization() {
        use chrono::TimeZone;
        let mut data = InverterData::default();
        let now = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();
        data.timestamp = now;
        data.w = 1000.0;
        data.wh = 5000.0;
        data.st = "MPPT".to_string();
        data.v_an = None; // Explicitly None

        let json = serde_json::to_string(&data).unwrap();
        // chrono's rfc3339 might differ slightly from serde's default,
        // but for a whole second they should match.
        assert!(json.contains("\"timestamp\":\"2024-01-01T12:00:00Z\""));
        assert!(json.contains("\"W\":1000.0"));
        assert!(json.contains("\"WH\":5000.0"));
        assert!(json.contains("\"St\":\"MPPT\""));
        assert!(json.contains("\"PhVphA\":null")); // Verify explicit null
    }
}
