# pv2mqtt

## Description

Bridge SunSpec-compliant inverters to MQTT, with Home Assistant autodiscovery.

One `pv2mqtt` instance can poll multiple devices on multiple buses (Modbus-TCP or
Modbus-RTU) and the refresh interval is configurable per device.

## Features

- Home Assistant integration: Inverters are registered with Home Assistant using MQTT discovery
- Security: Supports TLS (including mTLS) for both MQTT and Modbus-TCP.
- Robust: Automatic reconnection and efficient multi-device polling.
- Controls: Set maximum output power limit.

## Quick Start

1. Copy the configuration example, and adjust to your environment:
   ```
   cp pv2mqtt.toml.example pv2mqtt.toml
   ```
   This file also contains full documentation of configuration options.
2. Run (Docker):
   ```
   docker run --rm -v $(pwd)/pv2mqtt.toml:/app/pv2mqtt.toml:ro \
       ghcr.io/martijnvds/pv2mqtt:latest
   ```
   For Modbus-RTU, add `--device=/dev/ttyUSB0:/dev/ttyUSB0:rw` (change
   device name as needed).
3. Run (Cargo):
   ```
   cargo run --release -- [config_path]
   ```

If no configuration filename is specified, `/etc/pv2mqtt.toml` is used.

### Environment variables

MQTT credentials can be overridden using environment variables:

- `MQTT_USERNAME`, `MQTT_PASSWORD`
- `MQTT_USERNAME_FILE`, `MQTT_PASSWORD_FILE` - for Docker/K8s secrets

## Security & Safety

- Network Isolation: Modbus lacks authentication. Ensure that inverters
  are on a dedicated VLAN that only allows traffic from trusted sources.
- Encryption: Enable (m)TLS for MQTT and Modbus-TCP when supported.
- Controls: Disabled by default, enable using `enable_controls = true`
  - Changing the controls _will_ affect your energy production.

## Commands

When commands are enabled, and the inverter supports them, values can be
set by publishing to
`pv2mqtt/inverter/{serial}/set/{control}`, with `control` being
one of:

- `Conn` (on/off)
- `WMaxLim_Ena` (on/off)
- `WMaxLimPct` (percentage 0-100)

## Tested devices

The program has been tested with the following devices:

| Device            | Type            | Port (default) | Comment                                                                                                        |
| :---------------- | :-------------- | :------------: | -------------------------------------------------------------------------------------------------------------- |
| AP Systems YC600  | Micro inverter  |      502       | Works (via ECU-R-Pro); make sure Modbus is enabled                                                             |
| SolarEdge SE3680H | String inverter |      1502      | Works over wifi, untested over wired ethernet and RTU; for wifi/ethernet usage, Modbus-TCP needs to be enabled |

## Links

- [SunSpec Alliance](https://sunspec.org/) - Official SunSpec specifications.
- [sunspec](https://github.com/bikeshedder/sunspec) - The SunSpec library used by this project.
- [Home Assistant SunSpec integration](https://github.com/CJNE/ha-sunspec) - Alternative if all your inverters support Modbus-TCP and you only need your data in Home Assistant.
