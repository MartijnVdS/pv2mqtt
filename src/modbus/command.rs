// SPDX-License-Identifier: Apache-2.0

use super::{
    COMMAND_POLL_DELAY_MILLIS, ConnectionTask, DeviceState, InverterConnection, M123_CONN_OFFSET,
    M123_WMAX_LIM_ENA_OFFSET, M123_WMAX_LIM_PCT_OFFSET, M123_WMAX_LIM_PCT_SF_OFFSET,
    M704_WMAX_LIM_ENA_OFFSET, M704_WMAX_LIM_PCT_OFFSET, M704_WMAX_LIM_PCT_SF_OFFSET,
};
use crate::commands::ControlAction;
use crate::error::{ModbusError, Pv2MqttError, Result};
use crate::models::ActiveControlModel;
use std::time::Duration;
use tracing::{debug, error, info, warn};

impl<C: InverterConnection> ConnectionTask<C> {
    pub async fn handle_command(
        devices: &mut [DeviceState<C>],
        cmd: crate::commands::ModbusCommand,
    ) {
        // Find if this command is for one of our devices
        let device_state = match devices
            .iter_mut()
            .find(|d| d.serial.as_ref() == Some(&cmd.serial))
        {
            Some(d) => d,
            None => return, // Not for us
        };

        if !device_state.config.enable_controls {
            warn!(
                "Received command for device {}, but controls are not enabled in config",
                cmd.serial
            );
            return;
        }

        let unit_id = device_state.config.unit_id;
        let action = cmd.action;

        info!("Executing command for unit {}: {:?}", unit_id, action);

        let device = match &device_state.device {
            Some(d) => d,
            None => {
                error!(
                    "Cannot execute command for unit {}: device not connected",
                    unit_id
                );
                return;
            }
        };

        let res: Result<()> = async {
            let (base_addr, conn_off, pct_off, ena_off, sf_off) = match device_state.active_control
            {
                ActiveControlModel::Model123 { base_addr } => (
                    base_addr,
                    Some(M123_CONN_OFFSET),
                    M123_WMAX_LIM_PCT_OFFSET,
                    M123_WMAX_LIM_ENA_OFFSET,
                    M123_WMAX_LIM_PCT_SF_OFFSET,
                ),
                ActiveControlModel::Model704 { base_addr } => (
                    base_addr,
                    None,
                    M704_WMAX_LIM_PCT_OFFSET,
                    M704_WMAX_LIM_ENA_OFFSET,
                    M704_WMAX_LIM_PCT_SF_OFFSET,
                ),
                ActiveControlModel::None => {
                    return Err(Pv2MqttError::Modbus(ModbusError::from(
                        std::io::Error::other("Controls not supported or identified for device"),
                    )));
                }
            };

            match action {
                ControlAction::Conn(connect) => {
                    Self::write_conn(device.as_ref(), base_addr, conn_off, connect).await
                }
                ControlAction::WMaxLimPct(pct) => {
                    Self::write_wmax_lim_pct(device.as_ref(), base_addr, pct_off, sf_off, pct).await
                }
                ControlAction::WMaxLimEna(enable) => {
                    Self::write_wmax_lim_ena(device.as_ref(), base_addr, ena_off, enable).await
                }
            }
        }
        .await;

        if let Err(e) = res {
            error!("Failed to execute command for {}: {}", cmd.serial, e);
        } else {
            info!(
                "Successfully executed command for {}. Waiting {}ms before immediate poll.",
                cmd.serial, COMMAND_POLL_DELAY_MILLIS
            );
        }

        // Small delay to allow hardware to settle and update internal state
        tokio::time::sleep(Duration::from_millis(COMMAND_POLL_DELAY_MILLIS)).await;
        device_state.last_poll = None;
    }

    async fn write_conn(
        device: &C,
        base_addr: u16,
        conn_off: Option<u16>,
        connect: bool,
    ) -> Result<()> {
        let offset = conn_off.ok_or_else(|| {
            Pv2MqttError::Modbus(ModbusError::from(std::io::Error::other(
                "Connect/Disconnect command not supported by this control model",
            )))
        })?;
        let val = if connect { 1 } else { 0 };
        debug!(
            "Writing Conn={} (val={}) to address {}",
            connect,
            val,
            base_addr + offset
        );
        device.write_registers(base_addr + offset, &[val]).await
    }

    async fn write_wmax_lim_pct(
        device: &C,
        base_addr: u16,
        pct_off: u16,
        sf_off: u16,
        pct: f32,
    ) -> Result<()> {
        let regs = device.read_registers(base_addr + sf_off, 1).await?;
        let sf = regs[0] as i16;

        // Calculate raw value: raw = pct / 10^sf
        let factor = 10f32.powi(sf as i32);
        let raw = (pct / factor).round() as u16;

        debug!(
            "Writing WMaxLimPct={} (raw={}) to address {}",
            pct,
            raw,
            base_addr + pct_off
        );
        device.write_registers(base_addr + pct_off, &[raw]).await
    }

    async fn write_wmax_lim_ena(
        device: &C,
        base_addr: u16,
        ena_off: u16,
        enable: bool,
    ) -> Result<()> {
        let val = if enable { 1 } else { 0 };
        debug!(
            "Writing WMaxLim_Ena={} (val={}) to address {}",
            enable,
            val,
            base_addr + ena_off
        );
        device.write_registers(base_addr + ena_off, &[val]).await
    }
}
