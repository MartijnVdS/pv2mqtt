// SPDX-License-Identifier: Apache-2.0

pub mod m101;
pub mod m102;
pub mod m103;
pub mod m111;
pub mod m112;
pub mod m113;
pub mod m123;
pub mod traits;
pub mod types;

pub use traits::*;
pub use types::*;

use crate::error::{Pv2MqttError, Result};
use std::sync::Arc;
use sunspec::client::AsyncDevice;
use tokio::sync::Mutex;
use tokio_modbus::client::Context as ModbusContext;

pub static SUPPORTED_MODELS: &[u16] = &[101, 102, 103, 111, 112, 113, 123];

pub async fn poll_and_apply(
    model_id: u16,
    device: &AsyncDevice<Arc<Mutex<ModbusContext>>>,
    data: &mut InverterData,
) -> Result<()> {
    match model_id {
        101 => {
            let m = device
                .read_model::<sunspec::models::model101::Model101>()
                .await?;
            m.into_inverter_data(data);
        }
        102 => {
            let m = device
                .read_model::<sunspec::models::model102::Model102>()
                .await?;
            m.into_inverter_data(data);
        }
        103 => {
            let m = device
                .read_model::<sunspec::models::model103::Model103>()
                .await?;
            m.into_inverter_data(data);
        }
        111 => {
            let m = device
                .read_model::<sunspec::models::model111::Model111>()
                .await?;
            m.into_inverter_data(data);
        }
        112 => {
            let m = device
                .read_model::<sunspec::models::model112::Model112>()
                .await?;
            m.into_inverter_data(data);
        }
        113 => {
            let m = device
                .read_model::<sunspec::models::model113::Model113>()
                .await?;
            m.into_inverter_data(data);
        }
        123 => {
            let m = device
                .read_model::<sunspec::models::model123::Model123>()
                .await?;
            m.into_inverter_data(data);
        }
        _ => return Err(Pv2MqttError::UnsupportedModel(model_id)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::traits::SunSpecModel;
    use super::types::InverterData;
    use approx::assert_relative_eq;
    use sunspec::models::model101::{
        Evt1 as Evt1_101, Evt2 as Evt2_101, EvtVnd1 as EvtVnd1_101, EvtVnd2 as EvtVnd2_101,
        EvtVnd3 as EvtVnd3_101, EvtVnd4 as EvtVnd4_101, Model101, St as St101,
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

    #[test]
    fn test_model101_conversion() {
        let mut m101 = empty_m101();
        m101.ph_vph_a = 2300;
        m101.v_sf = -1;
        m101.w = 2300;
        m101.w_sf = 1;
        m101.st = St101::Mppt;
        let mut data = InverterData::default();
        m101.into_inverter_data(&mut data);
        assert_relative_eq!(data.v_an.unwrap(), 230.0);
        assert_relative_eq!(data.w, 23000.0);
        assert_eq!(data.st, "MPPT");
    }

    #[test]
    fn test_model101_large_energy_precision() {
        let mut m101 = empty_m101();
        m101.wh = 20000001;
        m101.wh_sf = 0;
        let mut data = InverterData::default();
        m101.into_inverter_data(&mut data);
        assert_eq!(data.wh, 20000001.0);
    }
}
