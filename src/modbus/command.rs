use super::{
    COMMAND_POLL_DELAY_MILLIS, ConnectionTask, DeviceState, M123_CONN_OFFSET,
    M123_WMAX_LIM_ENA_OFFSET, M123_WMAX_LIM_PCT_OFFSET, M123_WMAX_LIM_PCT_SF_OFFSET,
};
use crate::error::{ModbusError, Pv2MqttError, Result};
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

        let device = match &device_state.device {
            Some(d) => d,
            None => {
                warn!(
                    "Received command for device {}, but it is not currently connected",
                    cmd.serial
                );
                return;
            }
        };

        let unit_id = device_state.config.unit_id;
        let action = cmd.action;

        info!("Executing command for unit {}: {:?}", unit_id, action);

        let res: Result<()> = async {
            use crate::commands::ControlAction;

            if device.models.m123.addr == 0 {
                return Err(Pv2MqttError::Modbus(ModbusError::from(
                    std::io::Error::other("Model 123 not supported by device"),
                )));
            }

            let base_addr = device.models.m123.addr;
            let mut modbus = ctx.lock().await;
            modbus.set_slave(Slave(unit_id));

            match action {
                ControlAction::Conn(connect) => {
                    let val = if connect { 1 } else { 0 };
                    debug!(
                        "Writing Conn={} (val={}) to address {}",
                        connect,
                        val,
                        base_addr + M123_CONN_OFFSET
                    );
                    modbus
                        .write_multiple_registers(base_addr + M123_CONN_OFFSET, &[val])
                        .await
                        .map_err(ModbusError::from)?
                        .map_err(|e| {
                            ModbusError::Protocol(format!("Modbus exception writing Conn: {}", e))
                        })?;
                }
                ControlAction::WMaxLimPct(pct) => {
                    let regs = modbus
                        .read_holding_registers(base_addr + M123_WMAX_LIM_PCT_SF_OFFSET, 1)
                        .await
                        .map_err(ModbusError::from)?
                        .map_err(|e| {
                            ModbusError::Protocol(format!("Modbus exception reading SF: {}", e))
                        })?;
                    let sf = regs[0] as i16;

                    // Calculate raw value: raw = pct / 10^sf
                    let factor = 10f32.powi(sf as i32);
                    let raw = (pct / factor).round() as u16;

                    debug!(
                        "Writing WMaxLimPct={} (raw={}) to address {}",
                        pct,
                        raw,
                        base_addr + M123_WMAX_LIM_PCT_OFFSET
                    );
                    modbus
                        .write_multiple_registers(base_addr + M123_WMAX_LIM_PCT_OFFSET, &[raw])
                        .await
                        .map_err(ModbusError::from)?
                        .map_err(|e| {
                            ModbusError::Protocol(format!(
                                "Modbus exception writing WMaxLimPct: {}",
                                e
                            ))
                        })?;
                }
                ControlAction::WMaxLimEna(enable) => {
                    let val = if enable { 1 } else { 0 };
                    debug!(
                        "Writing WMaxLim_Ena={} (val={}) to address {}",
                        enable,
                        val,
                        base_addr + M123_WMAX_LIM_ENA_OFFSET
                    );
                    modbus
                        .write_multiple_registers(base_addr + M123_WMAX_LIM_ENA_OFFSET, &[val])
                        .await
                        .map_err(ModbusError::from)?
                        .map_err(|e| {
                            ModbusError::Protocol(format!(
                                "Modbus exception writing WMaxLim_Ena: {}",
                                e
                            ))
                        })?;
                }
            }
            Ok(())
        }
        .await;

        if let Err(e) = res {
            error!("Failed to execute command for {}: {}", cmd.serial, e);
        } else {
            info!(
                "Successfully executed command for {}. Waiting {}ms before immediate poll.",
                cmd.serial, COMMAND_POLL_DELAY_MILLIS
            );
            // Small delay to allow hardware to settle and update internal state
            tokio::time::sleep(Duration::from_millis(COMMAND_POLL_DELAY_MILLIS)).await;
            device_state.last_poll = None;
        }
    }
}
