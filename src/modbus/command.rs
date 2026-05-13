use super::{
    COMMAND_POLL_DELAY_MILLIS, ConnectionTask, DeviceState, M123_CONN_OFFSET,
    M123_WMAX_LIM_ENA_OFFSET, M123_WMAX_LIM_PCT_OFFSET, M123_WMAX_LIM_PCT_SF_OFFSET,
    M704_WMAX_LIM_ENA_OFFSET, M704_WMAX_LIM_PCT_OFFSET, M704_WMAX_LIM_PCT_SF_OFFSET,
};
use crate::error::{ModbusError, Pv2MqttError, Result};
use crate::models::ActiveControlModel;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_modbus::Slave;
use tokio_modbus::client::{Context as ModbusContext, Reader, Writer};
use tokio_modbus::slave::SlaveContext;
use tracing::{debug, error, info, warn};

impl ConnectionTask {
    pub async fn handle_command(
        &self,
        ctx: &Arc<Mutex<ModbusContext>>,
        devices: &mut [DeviceState],
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

        let res: Result<()> = async {
            use crate::commands::ControlAction;

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

            let mut modbus = ctx.lock().await;
            modbus.set_slave(Slave(unit_id));

            match action {
                ControlAction::Conn(connect) => {
                    self.write_conn(&mut modbus, base_addr, conn_off, connect)
                        .await
                }
                ControlAction::WMaxLimPct(pct) => {
                    self.write_wmax_lim_pct(&mut modbus, base_addr, pct_off, sf_off, pct)
                        .await
                }
                ControlAction::WMaxLimEna(enable) => {
                    self.write_wmax_lim_ena(&mut modbus, base_addr, ena_off, enable)
                        .await
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
        &self,
        modbus: &mut ModbusContext,
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
        modbus
            .write_multiple_registers(base_addr + offset, &[val])
            .await
            .map_err(ModbusError::from)?
            .map_err(|e| ModbusError::Protocol(format!("Modbus exception writing Conn: {}", e)))?;
        Ok(())
    }

    async fn write_wmax_lim_pct(
        &self,
        modbus: &mut ModbusContext,
        base_addr: u16,
        pct_off: u16,
        sf_off: u16,
        pct: f32,
    ) -> Result<()> {
        let regs = modbus
            .read_holding_registers(base_addr + sf_off, 1)
            .await
            .map_err(ModbusError::from)?
            .map_err(|e| ModbusError::Protocol(format!("Modbus exception reading SF: {}", e)))?;
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
        modbus
            .write_multiple_registers(base_addr + pct_off, &[raw])
            .await
            .map_err(ModbusError::from)?
            .map_err(|e| {
                ModbusError::Protocol(format!("Modbus exception writing WMaxLimPct: {}", e))
            })?;
        Ok(())
    }

    async fn write_wmax_lim_ena(
        &self,
        modbus: &mut ModbusContext,
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
        modbus
            .write_multiple_registers(base_addr + ena_off, &[val])
            .await
            .map_err(ModbusError::from)?
            .map_err(|e| {
                ModbusError::Protocol(format!("Modbus exception writing WMaxLim_Ena: {}", e))
            })?;
        Ok(())
    }
}
