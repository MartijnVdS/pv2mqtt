# pv2mqtt 

## Description

Publish data from SunSpec-compliant inverters to an MQTT broker, including
Home Assistant autodiscovery.

This service is written in **Rust** and uses asynchronous communication to
efficiently poll multiple inverters.

It allows you to have all of the data of your solar (PV) inverters together in
one place on a local machine, instead of using a different API (and possibly
cloud services) for each one. It also allows multiple readers to use the
data without conflict.

One `pv2mqtt` instance can poll multiple devices on multiple buses (Modbus-TCP or
Modbus-RTU) and the refresh interval is configurable per device.

## Configuration

To configure `pv2mqtt`, make a copy of `pv2mqtt.toml.example` to `pv2mqtt.toml`
and edit it to match your setup.

```shell
cp pv2mqtt.toml.example pv2mqtt.toml
```

The configuration file uses the [TOML](https://toml.io/) format and contains:
* MQTT settings: Broker URL, topic prefixes for data and Home Assistant discovery.
* Connections: One or more Modbus connections (TCP or RTU).
* Devices: One or more SunSpec devices per Connection, each with its own
  Unit ID and polling interval.

### Environment Variables and Secrets

For secure deployments (e.g., Docker Secrets), you can provide MQTT credentials
via environment variables. These will overwrite any credentials specified in the
MQTT URL in the configuration file.

* `MQTT_USERNAME`: MQTT username.
* `MQTT_PASSWORD`: MQTT password.
* `MQTT_USERNAME_FILE`: Path to a file containing the MQTT username.
* `MQTT_PASSWORD_FILE`: Path to a file containing the MQTT password.

File-based variables (`_FILE`) take precedence over direct environment variables.

The example configuration file contains documentation in the form of comments.

## Building and Running

### Using Docker (Recommended)

The easiest way to run `pv2mqtt` is to use a container.

```shell
# Build the image
docker build -t pv2mqtt .

# Run the container, mounting your config file
docker run --rm \
    -v $(pwd)/pv2mqtt.toml:/app/pv2mqtt.toml:ro \
    pv2mqtt
```

If you use a serial (RS-485) connection, you also need to pass through the
serial device to the container:

```shell
docker run --rm \
    -v $(pwd)/pv2mqtt.toml:/app/pv2mqtt.toml:ro \
    --device=/dev/ttyUSB0:/dev/ttyUSB0:rw \
    pv2mqtt
```

### Using Cargo

If you have Rust installed, you can build and run it directly:

```shell
# Build and run
cargo run --release -- [config_file]
```

Note: If no `config_file` is specified, `pv2mqtt` defaults to `/etc/pv2mqtt.toml`.

## Security Best Practices

- Use TLS/mTLS: Always enable TLS for MQTT and Modbus-TCP when supported.
  `pv2mqtt` supports Mutual TLS (mTLS) for both by providing `cert_path` and
  `key_path` in the configuration.
- TLS Certificate Verification: `pv2mqtt` uses the system's root certificate
  store to verify server certificates. If you specify a path to your CA
  certificate using the `ca_path` option, only the certificates in that
  specific file will be trusted. This is recommended for high-security
  environments or when using self-signed certificates.
- Network Isolation: Most inverters use the standard Modbus protocol, which
  lacks built-in authentication. Even when using TLS, an attacker on your local
  network could bypass `pv2mqtt` and communicate directly with your hardware.
  *Always* isolate your inverters on a dedicated, firewalled network segment
  (VLAN) that only allows traffic from the machine running `pv2mqtt`.
- Principle of Least Privilege: Run `pv2mqtt` with the minimum necessary
  permissions. If using RTU, the user only needs access to the specific serial
  device.

## Bidirectional Controls (SunSpec Model 123)

`pv2mqtt` supports bidirectional control of inverters that implement SunSpec Model 123 (Immediate Controls). This feature is disabled by default for safety and must be explicitly enabled per device in your configuration.

### Enabling Controls
In your `pv2mqtt.toml`, set `enable_controls = true` for the specific device:

```toml
[[connections.devices]]
unit_id = 1
interval = 30
enable_controls = true
```

### Command Topics
When enabled, you can send commands to the inverter by publishing to the
following MQTT topics:

- `pv2mqtt/inverter/{serial}/set/Conn`: Connect/Disconnect the inverter.
  - Payloads: `ON`/`OFF`, `true`/`false`, or `1`/`0`.
- `pv2mqtt/inverter/{serial}/set/WMaxLimPct`: Set the maximum active power as a
  percentage of `WMax`.
  - Payload: A float value (e.g., `75.5`).
- `pv2mqtt/inverter/{serial}/set/WMaxLim_Ena`: Enable/Disable the power limit.
  - Payloads: `ON`/`OFF`, `true`/`false`, or `1`/`0`.

### Safety & Security Warnings
- **Experimental Feature**: Bidirectional control can affect your energy
  production (that's what it's for after all). Use with caution.
- Isolation: Because Modbus lacks authentication, enabling controls increases
  the importance of *Network Isolation*. Ensure your inverter cannot be reached
  by unauthorized devices on your network.

## Features

- *SunSpec Support*: Supports inverters that support the Sunspec protocol, and
  expose at least one of the following "models": 101, 102, 103, 111, 112, 113.
- *Home Assistant Discovery*: Automatically registers inverters in Home
  Assistant using MQTT autodiscovery.
- *JSON over MQTT*: Publishes data to MQTT in JSON format.
- *Security*: Supports TLS connections for MQTT and Modbus-TCP.
- *Robust*: Handles connection drops and reconnects automatically.

## Tested devices

The program has been tested with the following devices:

| Device | Type | Comment |
|-|-|-|
| AP Systems YC600 | Micro inverter with external monitoring box | Works (via ECU-R-Pro); make sure Modbus is enabled; default TCP port is 502 |
| SolarEdge SE3680H-RW000BNN4 | "Regular" inverter | Works over wifi, should also work over wired ethernet and RTU; for wifi/ethernet usage, Modbus-TCP needs to be enabled; default TCP port is 1502 |

## Links

* [SunSpec Alliance](https://sunspec.org/) - Official SunSpec specifications.
* [sunspec](https://github.com/bikeshedder/sunspec) - The SunSpec library used by this project.
* [Home Assistant SunSpec integration](https://github.com/CJNE/ha-sunspec) - Alternative if all your inverters support Modbus-TCP and you only need your data in Home Assistant.
