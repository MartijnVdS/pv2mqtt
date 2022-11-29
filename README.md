# pv2mqtt 

## Description

Publish data from SunSpec-compliant inverters to an MQTT broker, including
Home Assistant MQTT discovery.

This service allows me to have all of the data of my PV inverters together in
once place on a local machine, instead of using a different API (and possibly
cloud service) for each one. It also allows multiple "readers" to use the data
without conflict.

One `pv2mqtt` instance can poll multiple devices on multiple buses and the
refresh interval is configurable per device.

## Configuration

To configure `pv2mqtt`, make a copy of `pv2mqtt.dist.yml` and edit the copy
to match your setup.

You need to define a "connection" for each physical connection (= modbus bus)
your devices are connected to and a "device" for each device you want to read
data from.

In addition, you may need to set up a user account on your MQTT broker.

## Building/installing/running

The easiest way to run `pv2pqtt` is to use a container:

```shell
$ docker pull martijnvds/pv2mqtt:latest
$ # (images are available for amd64 and arm64)

$ # Or build it yourself:
$ docker build -t pv2mqtt .
```

This way, you will always have a supported Python version, and you won't clutter
up the system with dependencies.

It's also possible to install the dependencies manually, outside of a container
by using the included  `requirements.txt`.

Once you've downloaded or built the container, you can run it:

```shell
$ docker run --rm \
    --volume $(pwd)/pv2mqtt.yml:/pv2mqtt.yml:ro \
    martijnvds/pv2mqtt:latest
```

This makes the configuration file available in the container as `/pv2mqtt.yml`.
The container is built with to automatically start `pv2mqtt` using that
configuration file. You can specify a different one on the command line:

```shell
$ docker run --rm \
    --volume $(pwd)/pv2mqtt.yml:/etc/pv2mqtt.yml:ro \
    martijnvds/pv2mqtt:latest \
    /etc/pv2mqtt.yml
```

If you use a serial (RS-485) connection, you also need to pass through the
serial device to the container at startup. Make sure you use the same device
name in your configuration file!

```shell
$ docker run --rm \
    --volume=$(pwd)/pv2mqtt.yml:/pv2mqtt.yml:ro \
    --device=/dev/ttyUSB0:/dev/ttyUSB0:rw \
    martijnvds/pv2mqtt:latest
```

## Limitations

Currently, only inverter data is read and published.

## Tested devices

The program has been tested with the following devices:

| Device | Type | Comment |
|-|-|-|
| AP Systems YC600 | Micro inverter | Works (via ECU-R-Pro) |

## Links

* [Docker hub](https://hub.docker.com/r/martijnvds/pv2mqtt) - Container images
* [pysunspec2](https://github.com/sunspec/pysunspec2) - SunSpec library that does the heavy lifting.
* [Home Assistant SunSpec integration](https://github.com/CJNE/ha-sunspec) - Alternative if all your inverters support Modbus-TCP and you only need your data in Home Assistant.