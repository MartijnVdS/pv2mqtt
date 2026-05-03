// SPDX-License-Identifier: Apache-2.0

use super::types::InverterData;

pub trait SunSpecModel {
    /// Converts the parsed model data into the unified InverterData struct
    fn into_inverter_data(self, data: &mut InverterData);
}

pub trait ToStatusString {
    fn to_status_string(&self) -> String;
}
