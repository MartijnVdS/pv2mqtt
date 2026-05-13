// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Utc};
use serde::Serialize;

pub const SUNSPEC_UNIMPLEMENTED_U64: u64 = 0xFFFFFFFFFFFFFFFF;
pub const SUNSPEC_UNIMPLEMENTED_U32: u32 = 0xFFFFFFFF;
pub const SUNSPEC_UNIMPLEMENTED_U16: u16 = 0xFFFF;
pub const SUNSPEC_UNIMPLEMENTED_I16: i16 = -32768;

#[derive(Debug, Serialize, Default, Clone, Copy, PartialEq, Eq)]
pub enum ActiveControlModel {
    #[default]
    None,
    Model123 {
        base_addr: u16,
    },
    Model704 {
        base_addr: u16,
    },
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct ControlData {
    #[serde(rename = "Conn")]
    pub conn: Option<bool>,
    #[serde(rename = "WMaxLimPct")]
    pub w_max_lim_pct: Option<f32>,
    #[serde(rename = "WMaxLim_Ena")]
    pub w_max_lim_ena: Option<bool>,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct NameplateData {
    #[serde(rename = "WMax")]
    pub w_max: Option<f32>,
    #[serde(rename = "VAMax")]
    pub va_max: Option<f32>,
    #[serde(rename = "VArMaxInj")]
    pub var_max_inj: Option<f32>,
    #[serde(rename = "VArMaxAbs")]
    pub var_max_abs: Option<f32>,
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
    #[serde(rename = "TmpAmb", skip_serializing_if = "Option::is_none")]
    pub tmp_amb: Option<f32>,
    #[serde(rename = "TmpSw", skip_serializing_if = "Option::is_none")]
    pub tmp_sw: Option<f32>,
    #[serde(rename = "Alrm", skip_serializing_if = "Option::is_none")]
    pub alrm: Option<Vec<String>>,
    #[serde(rename = "MnAlrmInfo", skip_serializing_if = "Option::is_none")]
    pub mn_alrm_info: Option<String>,
    #[serde(rename = "DERMode", skip_serializing_if = "Option::is_none")]
    pub der_mode: Option<Vec<String>>,
    #[serde(rename = "ConnSt", skip_serializing_if = "Option::is_none")]
    pub conn_st: Option<bool>,
    #[serde(rename = "ThrotPct", skip_serializing_if = "Option::is_none")]
    pub throt_pct: Option<f32>,
    #[serde(rename = "ThrotSrc", skip_serializing_if = "Option::is_none")]
    pub throt_src: Option<Vec<String>>,
    #[serde(rename = "St")]
    pub st: String,
    #[serde(rename = "Controls", skip_serializing_if = "Option::is_none")]
    pub controls: Option<ControlData>,
}

pub trait NumericValue {
    fn to_f64(self) -> Option<f64>;
}

impl NumericValue for u16 {
    fn to_f64(self) -> Option<f64> {
        if self == SUNSPEC_UNIMPLEMENTED_U16 {
            None
        } else {
            Some(self as f64)
        }
    }
}

impl NumericValue for i16 {
    fn to_f64(self) -> Option<f64> {
        if self == SUNSPEC_UNIMPLEMENTED_I16 {
            None
        } else {
            Some(self as f64)
        }
    }
}

impl NumericValue for u32 {
    fn to_f64(self) -> Option<f64> {
        if self == SUNSPEC_UNIMPLEMENTED_U32 {
            None
        } else {
            Some(self as f64)
        }
    }
}

impl NumericValue for u64 {
    fn to_f64(self) -> Option<f64> {
        if self == SUNSPEC_UNIMPLEMENTED_U64 {
            None
        } else {
            Some(self as f64)
        }
    }
}

impl<T: NumericValue> NumericValue for Option<T> {
    fn to_f64(self) -> Option<f64> {
        self.and_then(|v| v.to_f64())
    }
}

pub trait ScaleFactor {
    fn to_i32(self) -> Option<i32>;
}

impl ScaleFactor for i16 {
    fn to_i32(self) -> Option<i32> {
        Some(self as i32)
    }
}

impl ScaleFactor for Option<i16> {
    fn to_i32(self) -> Option<i32> {
        self.map(|s| s as i32)
    }
}

/// Applies a SunSpec scale factor to a numeric value, returning an f32.
pub fn apply_sf<V: NumericValue, S: ScaleFactor>(val: V, sf: S) -> Option<f32> {
    apply_sf_f64(val, sf).map(|v| v as f32)
}

/// Applies a SunSpec scale factor to a numeric value, returning an f64.
pub fn apply_sf_f64<V: NumericValue, S: ScaleFactor>(val: V, sf: S) -> Option<f64> {
    let v = val.to_f64()?;
    let s = sf.to_i32()?;
    Some(v * 10f64.powi(s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chrono::TimeZone;

    #[test]
    fn test_apply_sf() {
        assert_relative_eq!(apply_sf(100u16, 0i16).unwrap(), 100.0);
        assert_relative_eq!(apply_sf(100u16, -1i16).unwrap(), 10.0);
        assert_relative_eq!(apply_sf(100u16, 1i16).unwrap(), 1000.0);
        assert_eq!(apply_sf(SUNSPEC_UNIMPLEMENTED_U16, -1i16), None);
    }

    #[test]
    fn test_apply_sf_i16() {
        assert_relative_eq!(apply_sf(100i16, 0i16).unwrap(), 100.0);
        assert_relative_eq!(apply_sf(-100i16, -1i16).unwrap(), -10.0);
        assert_eq!(apply_sf(SUNSPEC_UNIMPLEMENTED_I16, -1i16), None);
    }

    #[test]
    fn test_apply_sf_i16_opt() {
        assert_relative_eq!(apply_sf(Some(100i16), Some(0i16)).unwrap(), 100.0);
        assert_relative_eq!(apply_sf(Some(100i16), Some(-1i16)).unwrap(), 10.0);
        assert_eq!(apply_sf(None::<i16>, Some(-1i16)), None);
        assert_eq!(apply_sf(Some(100i16), None::<i16>), None);
        assert_eq!(apply_sf(None::<i16>, None::<i16>), None);
        assert_eq!(apply_sf(Some(SUNSPEC_UNIMPLEMENTED_I16), Some(-1i16)), None);
    }

    #[test]
    fn test_apply_sf_u32_f64() {
        assert_relative_eq!(apply_sf_f64(100000u32, 0i16).unwrap(), 100000.0);
        assert_relative_eq!(apply_sf_f64(123456u32, -2i16).unwrap(), 1234.56);
        assert_eq!(apply_sf_f64(SUNSPEC_UNIMPLEMENTED_U32, -2i16), None);
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
