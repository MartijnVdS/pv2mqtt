// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone)]
pub enum ControlAction {
    Conn(bool),
    WMaxLimPct(f32),
    WMaxLimEna(bool),
}

#[derive(Debug, Clone)]
pub struct ModbusCommand {
    pub serial: String,
    pub action: ControlAction,
}
