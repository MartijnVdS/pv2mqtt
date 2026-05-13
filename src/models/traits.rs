// SPDX-License-Identifier: Apache-2.0

use super::types::InverterData;

pub trait SunSpecModel {
    /// Converts the parsed model data into the unified InverterData struct
    fn into_inverter_data(self, data: &mut InverterData);
}

pub trait ToStatusString {
    fn to_status_string(&self) -> String;
}

#[macro_export]
macro_rules! impl_to_status_string_for_st {
    ($t:ty, $off:ident, $sleeping:ident, $starting:ident, $mppt:ident, $throttled:ident, $shutting:ident, $fault:ident, $standby:ident) => {
        impl $crate::models::traits::ToStatusString for $t {
            fn to_status_string(&self) -> String {
                match self {
                    Self::$off => "OFF",
                    Self::$sleeping => "SLEEPING",
                    Self::$starting => "STARTING",
                    Self::$mppt => "MPPT",
                    Self::$throttled => "THROTTLED",
                    Self::$shutting => "SHUTTING_DOWN",
                    Self::$fault => "FAULT",
                    Self::$standby => "STANDBY",
                    _ => "UNKNOWN",
                }
                .to_string()
            }
        }
    };
}

#[macro_export]
macro_rules! map_sunspec_flags {
    ($val:expr, $type:ty, [ $($variant:ident => $string:expr),* $(,)? ]) => {
        {
            let mut flags = Vec::new();
            $(
                if $val.contains(<$type>::$variant) {
                    flags.push($string.to_string());
                }
            )*
            if flags.is_empty() { None } else { Some(flags) }
        }
    };
}
