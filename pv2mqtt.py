import abc
import dataclasses
import ipaddress
import logging
import paho.mqtt.client as mqtt_client
import pydantic
import pydantic.fields
import queue
import sunspec2.modbus.client as sunspec_client
import sunspec2.modbus.modbus as sunspec_modbus
import sys
import threading
import time
import yaml
from typing import Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class InverterData(pydantic.BaseModel):
    AphA: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Phase A current",
            "device_class": "current",
            "unit_of_measurement": "A",
            "state_class": "measurement",
            "value_template": "{{ value_json.AphA }}",
        },
    )
    AphB: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Phase B current",
            "device_class": "current",
            "unit_of_measurement": "A",
            "state_class": "measurement",
            "value_template": "{{ value_json.AphB }}",
        },
    )
    AphC: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Phase C current",
            "device_class": "current",
            "unit_of_measurement": "A",
            "state_class": "measurement",
            "value_template": "{{ value_json.AphC }}",
        },
    )
    PhVphA: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "AC Voltage AN",
            "device_class": "voltage",
            "unit_of_measurement": "V",
            "state_class": "measurement",
            "value_template": "{{ value_json.PhVphA }}",
        },
    )
    PhVphB: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "AC Voltage BN",
            "device_class": "voltage",
            "unit_of_measurement": "V",
            "state_class": "measurement",
            "value_template": "{{ value_json.PhVphB }}",
        },
    )
    PhVphC: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "AC Voltage CN",
            "device_class": "voltage",
            "unit_of_measurement": "V",
            "state_class": "measurement",
            "value_template": "{{ value_json.PhVphC }}",
        },
    )
    W: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Power",
            "enabled_by_default": True,
            "device_class": "power",
            "unit_of_measurement": "W",
            "state_class": "measurement",
            "value_template": "{{ value_json.W }}",
        },
    )
    VA: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Apparent power",
            "device_class": "apparent_power",
            "unit_of_measurement": "VA",
            "state_class": "measurement",
            "value_template": "{{ value_json.VA }}",
        },
    )
    VAr: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Reactive power",
            "device_class": "reactive_power",
            "unit_of_measurement": "var",
            "state_class": "measurement",
            "value_template": "{{ value_json.VAr }}",
        },
    )
    WH: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Energy",
            "enabled_by_default": True,
            "device_class": "energy",
            "unit_of_measurement": "kWh",
            "state_class": "total_increasing",
            "value_template": "{{ value_json.WH / 1000 | round(3) }}",
        },
    )
    PF: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Power factor (cos φ)",
            "device_class": "power_factor",
            "state_class": "measurement",
            "value_template": "{{ value_json.PF * 100 }}",
        },
    )
    Hz: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Grid frequency",
            "device_class": "frequency",
            "unit_of_measurement": "Hz",
            "state_class": "measurement",
            "value_template": "{{ value_json.Hz }}",
        },
    )
    TmpCab: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Cabinet temperature",
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "value_template": "{{ value_json.TmpCab }}",
        },
    )
    TmpSnk: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Heat sink temperature",
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "value_template": "{{ value_json.TmpSnk }}",
        },
    )
    TmpTrns: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Transformer temperature",
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "value_template": "{{ value_json.TmpTrns }}",
        },
    )
    TmpOt: float | None = pydantic.Field(
        None,
        json_schema_extra={
            "title": "Other temperature",
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "value_template": "{{ value_json.TmpOt }}",
        },
    )

    @classmethod
    def from_sunspec_model(
        cls, model: sunspec_client.SunSpecModbusClientModel
    ):
        return cls.model_validate(
            {
                field: getattr(model, field).cvalue
                for field in cls.model_fields.keys()
            }
        )


@dataclasses.dataclass
class Result:
    serial: str
    inverter_data: InverterData


class HADevice(pydantic.BaseModel):
    identifiers: list[str]
    manufacturer: str
    model: str
    name: str
    sw_version: str


class HADiscoveryData(pydantic.BaseModel):
    device: HADevice
    device_class: str
    enabled_by_default: bool
    force_update: bool
    name: str
    state_class: str
    state_topic: str
    unit_of_measurement: str
    unique_id: str
    value_template: str


class ModbusConnectionConfigBase(pydantic.BaseModel, abc.ABC):
    timeout_seconds: int = 10
    reuse_connection: bool = True

    device_type: Literal["inverter"] = pydantic.Field("inverter")

    @abc.abstractmethod
    def connect(
        self, device_id: int
    ) -> sunspec_client.SunSpecModbusClientDevice:
        pass


class ModbusTCPDeviceConfig(ModbusConnectionConfigBase):
    type: Literal["tcp"]
    ip_address: ipaddress.IPv4Address | ipaddress.IPv6Address
    port: int = 502

    def connect(
        self, device_id: int
    ) -> sunspec_client.SunSpecModbusClientDeviceTCP:
        return sunspec_client.SunSpecModbusClientDeviceTCP(
            slave_id=device_id,
            ipaddr=str(self.ip_address),
            ipport=self.port,
            timeout=self.timeout_seconds,
        )


class ModbusRTUDeviceConfig(ModbusConnectionConfigBase):
    type: Literal["rtu"]
    device: str
    baudrate: int = 9600
    parity: Literal["N"] | Literal["E"] = "N"

    def connect(
        self, device_id: int
    ) -> sunspec_client.SunSpecModbusClientDeviceRTU:
        return sunspec_client.SunSpecModbusClientDeviceRTU(
            slave_id=device_id,
            name=self.device,
            baudrate=self.baudrate,
            parity=self.parity,
            timeout=self.timeout_seconds,
        )


ModbusConnectionConfig = ModbusRTUDeviceConfig | ModbusTCPDeviceConfig


class DeviceConfig(pydantic.BaseModel):
    connection: str
    device_id: int = 1
    poll_interval_seconds: int = 300


class MQTTConfig(pydantic.BaseModel):
    host: str
    port: int = 1883

    username: None | str = None
    password: None | str = None

    topic_base: str = "pv2mqtt"
    discovery_base: str = "homeassistant"
    discovery_node: str = "node"


class Settings(pydantic.BaseModel):
    connections: dict[str, ModbusConnectionConfig]
    devices: list[DeviceConfig]
    mqtt_config: MQTTConfig


class SunSpecInverter:
    device: sunspec_client.SunSpecModbusClientDevice

    def __init__(self, device: sunspec_client.SunSpecModbusClientDevice):
        self.device = device

    def setup(self) -> None:
        self.device.scan()

        common_model = self.device.models["common"][0]
        self.manufacturer = common_model.Mn.value
        self.model = common_model.Md.value
        self.option = common_model.Opt.value
        self.version = common_model.Vr.value
        self.serial = common_model.SN.value

        for model_id in [101, 102, 103, 111, 112, 113]:
            # Pick the first inverter model that exists
            # 101, 102 and 103 contain integer data + scale factor (preferred)
            # 111, 112 and 113 contain floating point data
            # Some inverters have both, so we explicitly only use the first
            # one we find.
            if model_id in self.device.models:
                self.model_id = model_id
                break

    def connect(self) -> None:
        self.device.connect()

    def is_connected(self) -> bool:
        return self.device.is_connected()

    def disconnect(self) -> None:
        self.device.disconnect()

    def refresh(self) -> None:
        "Refresh the model's data."
        model = self.device.models[self.model_id][0]
        model.read()

        self.inverter_data = InverterData.from_sunspec_model(model)

    def _ha_device_meta(self) -> HADevice:
        return HADevice(
            identifiers=[self.serial],
            manufacturer=self.manufacturer,
            model=self.model,
            sw_version=self.version,
            name=f"Inverter {self.serial}",
        )

    def ha_discovery_data(
        self, state_topic: str
    ) -> dict[str, HADiscoveryData]:
        data: dict[str, HADiscoveryData] = {}
        device_meta = self._ha_device_meta()

        for field, field_model in InverterData.model_fields.items():
            extra = field_model.json_schema_extra
            assert isinstance(extra, dict)

            data[field] = HADiscoveryData(
                device=device_meta,
                device_class=extra["device_class"],
                enabled_by_default=extra.get("enabled_by_default", False),
                force_update=True,
                name=field_model.title or "",
                state_class=extra["state_class"],
                state_topic=state_topic,
                unit_of_measurement=extra.get("unit_of_measurement", " "),
                unique_id=f"pv2mqtt_{self.serial}_{field}",
                value_template=extra.get("value_template", ""),
            )

        return data


class MQTT:
    def __init__(self, config: MQTTConfig):
        self.config = config

    @staticmethod
    def on_mqtt_disconnect(client, userdata, rc):
        if rc != 0:
            logger.warning(
                "MQTT got disconnected. Will automatically reconnect."
            )

    def connect(self) -> None:
        mqtt = mqtt_client.Client()
        mqtt.enable_logger(logger)

        mqtt.on_disconnect = self.on_mqtt_disconnect

        if self.config.username:
            mqtt.username_pw_set(self.config.username, self.config.password)

        mqtt.connect(self.config.host, self.config.port)
        mqtt.loop_start()

        self.mqtt = mqtt

    def _data_topic(self, serial: str) -> str:
        topic_base = self.config.topic_base
        return f"{topic_base}/inverter/{serial}"

    def publish_data(self, serial: str, data: str) -> None:
        "Publish inverter data to MQTT"

        mqtt_topic = self._data_topic(serial)
        rv = self.mqtt.publish(mqtt_topic, data, qos=2)
        rv.wait_for_publish()

    def publish_discovery(
        self,
        inverter: SunSpecInverter,
    ) -> None:
        discovery_base = self.config.discovery_base

        discovery_data = inverter.ha_discovery_data(
            self._data_topic(inverter.serial)
        )

        for field, data in discovery_data.items():
            discovery_topic = (
                f"{discovery_base}/sensor/{inverter.serial}/{field}/config"
            )
            rv = self.mqtt.publish(
                discovery_topic, data.model_dump_json(), qos=2, retain=True
            )
            rv.wait_for_publish()


def load_config(config_file: str) -> Settings:
    with open(config_file, "rb") as cfg:
        raw_config = yaml.load(cfg, yaml.Loader)

    return Settings.model_validate(raw_config)


def run_polling_loop(
    lock: threading.Lock,
    result_queue: queue.Queue[Result],
    device: SunSpecInverter,
    poll_interval_seconds: int,
    reuse_connection: bool,
):
    logger.info(
        f"Starting polling loop: {device.serial} every {poll_interval_seconds}s"
    )

    while True:
        logger.info(
            "Refreshing data for "
            f"{device.manufacturer} {device.model} {device.serial}"
        )

        try:
            with lock:
                if not device.is_connected():
                    device.connect()

                device.refresh()
                inverter_data = device.inverter_data

                result_queue.put(
                    Result(serial=device.serial, inverter_data=inverter_data)
                )

                if not reuse_connection:
                    device.disconnect()

        except (ConnectionError, sunspec_modbus.ModbusClientError) as exc:
            device.disconnect()
            logger.warning(f"Error retrieving inverter data: {exc}")

        time.sleep(poll_interval_seconds)


def main(config: Settings):
    mqtt = MQTT(config.mqtt_config)
    mqtt.connect()

    result_queue: queue.Queue[Result] = queue.Queue()

    devices_by_connection: dict[str, list[DeviceConfig]] = {}
    for device_cfg in config.devices:
        if device_cfg.connection in devices_by_connection:
            devices_by_connection[device_cfg.connection].append(device_cfg)
        else:
            devices_by_connection[device_cfg.connection] = [device_cfg]

    polling_threads: list[threading.Thread] = []
    for connection_name, connection_cfg in config.connections.items():
        # Shared lock for all devices that use the same connection
        # This prevents concurrent access.
        lock = threading.Lock()

        for device_cfg in devices_by_connection[connection_name]:
            device = connection_cfg.connect(device_id=device_cfg.device_id)
            inverter = SunSpecInverter(device=device)
            inverter.setup()

            mqtt.publish_discovery(inverter)

            polling_thread = threading.Thread(
                target=run_polling_loop,
                daemon=True,
                kwargs={
                    "lock": lock,
                    "result_queue": result_queue,
                    "device": inverter,
                    "poll_interval_seconds": device_cfg.poll_interval_seconds,
                    "reuse_connection": connection_cfg.reuse_connection,
                },
            )
            polling_threads.append(polling_thread)

    for t in polling_threads:
        t.start()

    while queue_item := result_queue.get():
        logger.info(f"Publishing data for {queue_item.serial} to MQTT")
        mqtt.publish_data(
            queue_item.serial, queue_item.inverter_data.model_dump_json()
        )

        result_queue.task_done()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(f"Usage: {sys.argv[0]} config_file_name")

    main(load_config(sys.argv[1]))
