// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Utc};
use serde::Serialize;

pub const SUNSPEC_UNIMPLEMENTED_U32: u32 = 0xFFFFFFFF;
pub const SUNSPEC_UNIMPLEMENTED_U16: u16 = 0xFFFF;
pub const SUNSPEC_UNIMPLEMENTED_I16: i16 = -32768;

#[derive(Debug, Serialize, Default, Clone)]
pub struct Model123Data {
    #[serde(rename = "Conn")]
    pub conn: Option<bool>,
    #[serde(rename = "WMaxLimPct")]
    pub w_max_lim_pct: Option<f32>,
    #[serde(rename = "WMaxLim_Ena")]
    pub w_max_lim_ena: Option<bool>,
}

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
    pub wh: f64,
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
    #[serde(rename = "Controls", skip_serializing_if = "Option::is_none")]
    pub controls: Option<Model123Data>,
}

pub fn apply_sf(val: u16, sf: i16) -> Option<f32> {
    if val == SUNSPEC_UNIMPLEMENTED_U16 {
        return None;
    }
    Some(val as f32 * 10f32.powi(sf as i32))
}

pub fn apply_sf_i16(val: i16, sf: i16) -> Option<f32> {
    if val == SUNSPEC_UNIMPLEMENTED_I16 {
        return None;
    }
    Some(val as f32 * 10f32.powi(sf as i32))
}

pub fn apply_sf_u32_f64(val: u32, sf: i16) -> Option<f64> {
    if val == SUNSPEC_UNIMPLEMENTED_U32 {
        return None;
    }
    Some(val as f64 * 10f64.powi(sf as i32))
}

pub fn apply_sf_i16_opt(val: Option<i16>, sf: Option<i16>) -> Option<f32> {
    match (val, sf) {
        (Some(v), Some(s)) => apply_sf_i16(v, s),
        _ => None,
    }
}

pub fn apply_sf_opt(val: Option<u16>, sf: i16) -> Option<f32> {
    val.and_then(|v| apply_sf(v, sf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::TimeZone;

    #[test]
    fn test_apply_sf() {
        assert_relative_eq!(apply_sf(100, 0).unwrap(), 100.0);
        assert_relative_eq!(apply_sf(100, -1).unwrap(), 10.0);
        assert_relative_eq!(apply_sf(100, 1).unwrap(), 1000.0);
        assert_eq!(apply_sf(SUNSPEC_UNIMPLEMENTED_U16, -1), None);
    }

    #[test]
    fn test_apply_sf_i16() {
        assert_relative_eq!(apply_sf_i16(100, 0).unwrap(), 100.0);
        assert_relative_eq!(apply_sf_i16(-100, -1).unwrap(), -10.0);
        assert_eq!(apply_sf_i16(SUNSPEC_UNIMPLEMENTED_I16, -1), None);
    }

    #[test]
    fn test_apply_sf_i16_opt() {
        assert_relative_eq!(apply_sf_i16_opt(Some(100), Some(0)).unwrap(), 100.0);
        assert_relative_eq!(apply_sf_i16_opt(Some(100), Some(-1)).unwrap(), 10.0);
        assert_eq!(apply_sf_i16_opt(None, Some(-1)), None);
        assert_eq!(apply_sf_i16_opt(Some(100), None), None);
        assert_eq!(apply_sf_i16_opt(None, None), None);
        assert_eq!(
            apply_sf_i16_opt(Some(SUNSPEC_UNIMPLEMENTED_I16), Some(-1)),
            None
        );
    }

    #[test]
    fn test_apply_sf_u32_f64() {
        assert_relative_eq!(apply_sf_u32_f64(100000, 0).unwrap(), 100000.0);
        assert_relative_eq!(apply_sf_u32_f64(123456, -2).unwrap(), 1234.56);
        assert_eq!(apply_sf_u32_f64(SUNSPEC_UNIMPLEMENTED_U32, -2), None);
    }

    #[test]
    fn test_serialization() {
        let mut data = InverterData::default();
        let now = Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap();
        data.timestamp = now;
        data.w = 1000.0;
        data.wh = 5000.0;
        data.st = "MPPT".to_string();
        data.v_an = None;

        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("\"timestamp\":\"2024-01-01T12:00:00Z\""));
        assert!(json.contains("\"W\":1000.0"));
        assert!(json.contains("\"WH\":5000.0"));
        assert!(json.contains("\"St\":\"MPPT\""));
        assert!(json.contains("\"PhVphA\":null"));
    }
}
