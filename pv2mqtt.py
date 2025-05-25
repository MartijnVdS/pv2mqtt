import abc
import dataclasses
import ipaddress
import logging
import paho.mqtt.client as mqtt_client
import paho.mqtt.enums
import paho.mqtt.properties
import paho.mqtt.reasoncodes
import paho.mqtt.subscribeoptions
import pydantic
import queue
import sunspec2.modbus.client as sunspec_client
import sunspec2.modbus.modbus as sunspec_modbus
import sys
import threading
import time
import yaml
from typing import Any, ClassVar, Final, Literal, cast, override

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class InverterData(pydantic.BaseModel):
    AphA: float | None = pydantic.Field(
        None,
        title="Phase A current",
        json_schema_extra={
            "device_class": "current",
            "unit_of_measurement": "A",
            "state_class": "measurement",
            "value_template": "{{ value_json.AphA }}",
        },
    )
    AphB: float | None = pydantic.Field(
        None,
        title="Phase B current",
        json_schema_extra={
            "device_class": "current",
            "unit_of_measurement": "A",
            "state_class": "measurement",
            "value_template": "{{ value_json.AphB }}",
        },
    )
    AphC: float | None = pydantic.Field(
        None,
        title="Phase C current",
        json_schema_extra={
            "device_class": "current",
            "unit_of_measurement": "A",
            "state_class": "measurement",
            "value_template": "{{ value_json.AphC }}",
        },
    )
    PhVphA: float | None = pydantic.Field(
        None,
        title="AC Voltage AN",
        json_schema_extra={
            "device_class": "voltage",
            "unit_of_measurement": "V",
            "state_class": "measurement",
            "value_template": "{{ value_json.PhVphA }}",
        },
    )
    PhVphB: float | None = pydantic.Field(
        None,
        title="AC Voltage BN",
        json_schema_extra={
            "device_class": "voltage",
            "unit_of_measurement": "V",
            "state_class": "measurement",
            "value_template": "{{ value_json.PhVphB }}",
        },
    )
    PhVphC: float | None = pydantic.Field(
        None,
        title="AC Voltage CN",
        json_schema_extra={
            "device_class": "voltage",
            "unit_of_measurement": "V",
            "state_class": "measurement",
            "value_template": "{{ value_json.PhVphC }}",
        },
    )
    W: float | None = pydantic.Field(
        None,
        title="Power",
        json_schema_extra={
            "enabled_by_default": True,
            "device_class": "power",
            "unit_of_measurement": "W",
            "state_class": "measurement",
            "value_template": "{{ value_json.W }}",
        },
    )
    VA: float | None = pydantic.Field(
        None,
        title="Apparent power",
        json_schema_extra={
            "device_class": "apparent_power",
            "unit_of_measurement": "VA",
            "state_class": "measurement",
            "value_template": "{{ value_json.VA }}",
        },
    )
    VAr: float | None = pydantic.Field(
        None,
        title="Reactive power",
        json_schema_extra={
            "device_class": "reactive_power",
            "unit_of_measurement": "var",
            "state_class": "measurement",
            "value_template": "{{ value_json.VAr }}",
        },
    )
    WH: float | None = pydantic.Field(
        None,
        title="Energy",
        json_schema_extra={
            "enabled_by_default": True,
            "device_class": "energy",
            "unit_of_measurement": "kWh",
            "state_class": "total_increasing",
            "value_template": "{{ value_json.WH / 1000 | round(3) }}",
        },
    )
    PF: float | None = pydantic.Field(
        None,
        title="Power factor (cos φ)",
        json_schema_extra={
            "device_class": "power_factor",
            "state_class": "measurement",
            "value_template": "{{ value_json.PF * 100 }}",
            "unit_of_measurement": " ",
        },
    )
    Hz: float | None = pydantic.Field(
        None,
        title="Grid frequency",
        json_schema_extra={
            "device_class": "frequency",
            "unit_of_measurement": "Hz",
            "state_class": "measurement",
            "value_template": "{{ value_json.Hz }}",
        },
    )
    TmpCab: float | None = pydantic.Field(
        None,
        title="Cabinet temperature",
        json_schema_extra={
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "value_template": "{{ value_json.TmpCab }}",
        },
    )
    TmpSnk: float | None = pydantic.Field(
        None,
        title="Heat sink temperature",
        json_schema_extra={
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "value_template": "{{ value_json.TmpSnk }}",
        },
    )
    TmpTrns: float | None = pydantic.Field(
        None,
        title="Transformer temperature",
        json_schema_extra={
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "value_template": "{{ value_json.TmpTrns }}",
        },
    )
    TmpOt: float | None = pydantic.Field(
        None,
        title="Other temperature",
        json_schema_extra={
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "value_template": "{{ value_json.TmpOt }}",
        },
    )

    @classmethod
    def from_sunspec_model(cls, model: sunspec_client.SunSpecModbusClientModel):
        return cls.model_validate(
            {field: getattr(model, field).cvalue for field in cls.model_fields.keys()}
        )


class InverterControlData(pydantic.BaseModel):
    Conn: bool | None = pydantic.Field(
        default=None,
        title="Connection control",
        json_schema_extra={
            "type": "switch",
            "enabled_by_default": False,
        },
    )
    WMaxLimPct: float | None = pydantic.Field(
        default=None,
        title="Maximum Power Output limit",
        json_schema_extra={
            "type": "number",
            "unit_of_measurement": "%",
            "enabled_by_default": False,
            "min": 0,
            "max": 100,
        },
    )
    WMaxLim_Ena: bool | None = pydantic.Field(
        default=None,
        title="Maximum Power Output limit enabled",
        json_schema_extra={
            "type": "switch",
            "enabled_by_default": False,
        },
    )

    @classmethod
    def from_sunspec_model(cls, model: sunspec_client.SunSpecModbusClientModel):
        return cls.model_validate(
            {field: getattr(model, field).cvalue for field in cls.model_fields.keys()}
        )


@dataclasses.dataclass
class Result:
    serial: str
    inverter_data: InverterData
    inverter_control_data: InverterControlData | None


@dataclasses.dataclass
class Command:
    field: str
    value: bytes


class HADevice(pydantic.BaseModel):
    identifiers: list[str]
    manufacturer: str
    model: str
    name: str
    sw_version: str


class HASensorDiscoveryData(pydantic.BaseModel):
    entity_type: ClassVar[Final] = "sensor"
    device: HADevice
    name: str
    device_class: str
    enabled_by_default: bool
    force_update: bool
    state_class: str
    state_topic: str
    unit_of_measurement: str
    unique_id: str
    value_template: str


class HANumberDiscoveryData(pydantic.BaseModel):
    entity_type: ClassVar[Final] = "number"

    device: HADevice
    name: str
    command_topic: str
    enabled_by_default: bool
    min: float | None = None
    max: float | None = None
    retain: bool = False
    state_topic: str
    step: float = 0.1
    unit_of_measurement: str
    unique_id: str


class HASwitchDiscoveryData(pydantic.BaseModel):
    entity_type: ClassVar[Final] = "switch"

    device: HADevice
    name: str
    command_topic: str
    enabled_by_default: bool
    retain: bool = False
    state_topic: str
    unique_id: str


HADiscoveryData = HANumberDiscoveryData | HASensorDiscoveryData | HASwitchDiscoveryData


class ModbusConnectionConfigBase(pydantic.BaseModel, abc.ABC):
    timeout_seconds: int = 10
    reuse_connection: bool = True

    device_type: Literal["inverter"] = pydantic.Field("inverter")

    @abc.abstractmethod
    def connect(self, device_id: int) -> sunspec_client.SunSpecModbusClientDevice:
        pass


class ModbusTCPDeviceConfig(ModbusConnectionConfigBase):
    type: Literal["tcp"]
    ip_address: ipaddress.IPv4Address | ipaddress.IPv6Address
    port: int = 502

    @override
    def connect(self, device_id: int) -> sunspec_client.SunSpecModbusClientDeviceTCP:
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
    parity: Literal["N", "E"] = "N"

    @override
    def connect(self, device_id: int) -> sunspec_client.SunSpecModbusClientDeviceRTU:
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
    manufacturer: str
    model: str
    option: str
    version: str
    serial: str

    def __init__(self, device: sunspec_client.SunSpecModbusClientDevice):
        self.device = device

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
            # Some inverters have both integer and floating point models,
            # with the same data, so explicitly only use the first one found.
            if model_id in self.device.models:
                self.model_id: int = model_id
                break

        if 123 in self.device.models:
            # This inverter has "immediate controls"
            self.control_model_id: int | None = 123
        else:
            self.control_model_id = None

    def connect(self) -> None:
        self.device.connect()

    def disconnect(self) -> None:
        self.device.disconnect()

    def is_connected(self) -> bool:
        return self.device.is_connected()

    def refresh_inverter_data(self) -> InverterData:
        "Refresh the model's data."
        inverter_model = self.device.models[self.model_id][0]
        inverter_model.read()

        if self.control_model_id:
            control_model = self.device.models[self.control_model_id][0]
            control_model.read()
        else:
            control_model = None

        return InverterData.from_sunspec_model(inverter_model)

    def refresh_inverter_control_data(self) -> InverterControlData | None:
        "Refresh the 'inverter control' model data."
        if not self.control_model_id:
            return

        control_model = self.device.models[self.control_model_id][0]
        control_model.read()

        return InverterControlData.from_sunspec_model(control_model)

    def get_control_model(self) -> sunspec_client.SunSpecModbusClientModel | None:
        if not self.control_model_id:
            return

        control_model = self.device.models[self.control_model_id][0]
        control_model.read()

        return control_model

    def _ha_device_meta(self) -> HADevice:
        return HADevice(
            identifiers=[self.serial],
            manufacturer=self.manufacturer,
            model=self.model,
            sw_version=self.version,
            name=f"Inverter {self.serial}",
        )

    def ha_discovery_data(
        self,
        state_topic: str,
        control_topic_base: str,
    ) -> dict[str, HADiscoveryData]:
        data: dict[str, HADiscoveryData] = {}
        device_meta = self._ha_device_meta()

        for field, field_model in InverterData.model_fields.items():
            extra = field_model.json_schema_extra
            assert isinstance(extra, dict)

            data[field] = HASensorDiscoveryData(
                device=device_meta,
                device_class=cast(str, extra["device_class"]),
                enabled_by_default=cast(bool, extra.get("enabled_by_default", False)),
                force_update=True,
                name=field_model.title or "",
                state_class=cast(str, extra["state_class"]),
                state_topic=state_topic,
                unit_of_measurement=cast(str, extra.get("unit_of_measurement", " ")),
                unique_id=f"pv2mqtt_{self.serial}_{field}",
                value_template=cast(str, extra.get("value_template", "")),
            )

        for field, field_model in InverterControlData.model_fields.items():
            extra = field_model.json_schema_extra
            assert isinstance(extra, dict)

            match extra["type"]:
                case "switch":
                    data[field] = HASwitchDiscoveryData(
                        device=device_meta,
                        name=field_model.title or "",
                        command_topic=f"{control_topic_base}/{field}/set",
                        enabled_by_default=cast(
                            bool, extra.get("enabled_by_default", False)
                        ),
                        state_topic=f"{control_topic_base}/{field}",
                        unique_id=f"pv2mqtt_{self.serial}_control_{field}",
                    )
                case "number":
                    data[field] = HANumberDiscoveryData(
                        device=device_meta,
                        name=field_model.title or "",
                        command_topic=f"{control_topic_base}/{field}/set",
                        enabled_by_default=cast(
                            bool, extra.get("enabled_by_default", False)
                        ),
                        state_topic=f"{control_topic_base}/{field}",
                        unique_id=f"pv2mqtt_{self.serial}_control_{field}",
                        unit_of_measurement=cast(
                            str, extra.get("unit_of_measurement", "")
                        ),
                        min=cast(float, extra.get("min", 0)),
                        max=cast(float, extra.get("max", 1)),
                        # TODO: "step" value
                    )
                case _:
                    raise ValueError(f"Unknown type in control model: {extra['type']}")

        return data


class MQTT:
    def __init__(self, config: MQTTConfig) -> None:
        self.config = config
        self.command_queues = {}

        self.mqtt: mqtt_client.Client

    def register_command_queue(
        self,
        serial: str,
        command_queue: queue.Queue[Command],
    ) -> None:
        command_topic = self._control_topic_base(serial) + "/+/set"

        def send_command(
            client: mqtt_client.Client,
            userdata: Any,
            message: mqtt_client.MQTTMessage,
        ) -> None:
            topic_parts = message.topic.split("/")
            field = topic_parts[-2]

            command = Command(field=field, value=message.payload)
            command_queue.put(command)

        logger.debug(f"Adding callback for {command_topic}")
        self.mqtt.message_callback_add(sub=command_topic, callback=send_command)
        _ = self.mqtt.subscribe(
            topic=command_topic,
            options=paho.mqtt.subscribeoptions.SubscribeOptions(
                noLocal=True,
                qos=1,
            ),
        )

    @staticmethod
    def on_mqtt_disconnect(
        client: mqtt_client.Client,
        userdata: Any,
        disconnect_flags: mqtt_client.DisconnectFlags,
        rc: paho.mqtt.reasoncodes.ReasonCode,
        properties: paho.mqtt.properties.Properties | None,
    ) -> None:
        if rc != 0:
            logger.warning("MQTT got disconnected. Will automatically reconnect.")

    def connect(self) -> None:
        mqtt = mqtt_client.Client(
            callback_api_version=paho.mqtt.enums.CallbackAPIVersion.VERSION2,
            userdata=None,
            transport="tcp",
            clean_session=True,
        )
        mqtt.enable_logger(logger)

        mqtt.on_disconnect = self.on_mqtt_disconnect

        if self.config.username:
            mqtt.username_pw_set(self.config.username, self.config.password)

        _ = mqtt.connect(self.config.host, self.config.port)
        _ = mqtt.loop_start()

        self.mqtt = mqtt

    def _data_topic(self, serial: str) -> str:
        topic_base = self.config.topic_base
        return f"{topic_base}/inverter/{serial}"

    def _control_topic_base(self, serial: str) -> str:
        topic_base = self.config.topic_base
        return f"{topic_base}/inverter/{serial}/control"

    def publish_data(self, serial: str, data: str) -> None:
        "Publish inverter data to MQTT"

        mqtt_topic = self._data_topic(serial)
        rv = self.mqtt.publish(mqtt_topic, data, qos=2)
        rv.wait_for_publish()

    def publish_control(
        self,
        serial: str,
        inverter_control_data: InverterControlData,
    ) -> None:
        control_topic_base = self._control_topic_base(serial)
        logger.info(f"Publishing  control data {control_topic_base}")

        for field, _ in InverterControlData.model_fields.items():
            state_topic = f"{control_topic_base}/{field}"
            data = getattr(inverter_control_data, field)

            if isinstance(data, bool):
                data = "ON" if data else "OFF"

            rv = self.mqtt.publish(state_topic, payload=str(data), qos=2)
            rv.wait_for_publish()

    def publish_discovery(
        self,
        inverter: SunSpecInverter,
    ) -> None:
        discovery_base = self.config.discovery_base

        discovery_data = inverter.ha_discovery_data(
            state_topic=self._data_topic(inverter.serial),
            control_topic_base=self._control_topic_base(inverter.serial),
        )

        for field, data in discovery_data.items():
            discovery_topic = (
                f"{discovery_base}/{data.entity_type}/{inverter.serial}/{field}/config"
            )
            logger.debug(f"Publishing discovery metadata to {discovery_topic}")

            self.mqtt.publish(
                discovery_topic, data.model_dump_json(), qos=2, retain=True
            ).wait_for_publish()


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
    refresh_event: threading.Event,
):
    logger.info(
        f"Starting polling loop: {device.serial} every {poll_interval_seconds}s"
    )

    while True:
        logger.info(
            f"Refreshing data for {device.manufacturer} {device.model} {device.serial}"
        )

        with lock:
            try:
                if not device.is_connected():
                    device.connect()

                inverter_data = device.refresh_inverter_data()
                inverter_control_data = device.refresh_inverter_control_data()

                result_queue.put(
                    Result(
                        serial=device.serial,
                        inverter_data=inverter_data,
                        inverter_control_data=inverter_control_data,
                    )
                )

                if not reuse_connection:
                    device.disconnect()

            except (ConnectionError, sunspec_modbus.ModbusClientError) as exc:
                device.disconnect()
                logger.warning(f"Error retrieving inverter data: {exc}")

        if refresh_event.wait(poll_interval_seconds):
            refresh_event.clear()


def _parse_command_value(field_name: str, raw_value: bytes) -> float | int:
    field_definition = InverterControlData.model_fields[field_name].json_schema_extra
    assert isinstance(field_definition, dict)

    match field_definition["type"]:
        case "number":
            return float(raw_value)
        case "switch":
            if raw_value == b"ON":
                return 1
            elif raw_value == b"OFF":
                return 0
            else:
                raise ValueError("Invalid switch value (ON or OFF allowed)")
        case _:
            raise TypeError(f"Unknown type {field_definition['type']}")


def run_command_loop(
    lock: threading.Lock,
    command_queue: queue.Queue[Command],
    device: SunSpecInverter,
    reuse_connection: bool,
    refresh_event: threading.Event,
):
    logger.info(f"Starting command loop: {device.serial}")

    while command := command_queue.get():
        logger.info(f"Command for {device.serial}: {command.field}={command.value}")

        with lock:
            try:
                if not device.is_connected():
                    device.connect()

                control_model = device.get_control_model()
                if not control_model:
                    logger.info(
                        f"Inverter {device.serial} received a command, but inverter controls are not available?"
                    )
                    continue

                try:
                    parsed_value = _parse_command_value(
                        field_name=command.field, raw_value=command.value
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid command value: {e}")
                    continue

                getattr(control_model, command.field).cvalue = parsed_value
                control_model.write()

                logger.info(f"Set attribute: {command.field}={parsed_value}")

                # Trigger the refresh thread to reload inverter data:
                refresh_event.set()

                if not reuse_connection:
                    device.disconnect()
            except (ConnectionError, sunspec_modbus.ModbusClientError) as exc:
                device.disconnect()
                logger.warning(f"Error sending command: {exc}")


def main(config: Settings):
    result_queue: queue.Queue[Result] = queue.Queue()

    mqtt = MQTT(config.mqtt_config)
    mqtt.connect()

    devices_by_connection: dict[str, list[DeviceConfig]] = {}
    for device_cfg in config.devices:
        if device_cfg.connection in devices_by_connection:
            devices_by_connection[device_cfg.connection].append(device_cfg)
        else:
            devices_by_connection[device_cfg.connection] = [device_cfg]

    device_threads: list[threading.Thread] = []

    for connection_name, connection_cfg in config.connections.items():
        # Shared lock for all devices that use the same connection
        # This prevents concurrent access.
        lock = threading.Lock()

        for device_cfg in devices_by_connection[connection_name]:
            device = connection_cfg.connect(device_id=device_cfg.device_id)
            inverter = SunSpecInverter(device=device)

            command_queue: queue.Queue[Command] = queue.Queue()
            refresh_event = threading.Event()

            mqtt.register_command_queue(
                serial=inverter.serial,
                command_queue=command_queue,
            )

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
                    "refresh_event": refresh_event,
                },
            )
            device_threads.append(polling_thread)

            command_thread = threading.Thread(
                target=run_command_loop,
                daemon=True,
                kwargs={
                    "lock": lock,
                    "command_queue": command_queue,
                    "device": inverter,
                    "reuse_connection": connection_cfg.reuse_connection,
                    "refresh_event": refresh_event,
                },
            )
            device_threads.append(command_thread)

    for t in device_threads:
        t.start()

    while queue_item := result_queue.get():
        logger.info(f"Publishing data for {queue_item.serial} to MQTT")
        mqtt.publish_data(queue_item.serial, queue_item.inverter_data.model_dump_json())
        mqtt.publish_control(queue_item.serial, queue_item.inverter_control_data)

        result_queue.task_done()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(f"Usage: {sys.argv[0]} config_file_name")

    main(load_config(sys.argv[1]))
