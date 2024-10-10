from queue import Queue
import time
from algorithm import RaspberrySerialPort
from queue import Queue
from loguru import logger
from config import SerialCommConfig

logger.add(
    str(SerialCommConfig.getattr("temperature_data_save_path")),
    rotation="00:00", 
    filter=lambda record:record["extra"].get("name") == "temperature"
)
temperature_data_logger = logger.bind(name="temperature")

# 接收线程
def serial_receive(
        serial_ports: list[RaspberrySerialPort],
        queue: Queue,
        *args,
        **kwargs
):
    while True:
        for ser in serial_ports:
            ser.message_reception()
            temperature_data = ser.subcontracting()
            if temperature_data is not None:
                temperature_data_logger.info(temperature_data)
                queue.put(temperature_data)
        time.sleep(0.1)

# 发送线程
def serial_send(
    serial_ports: list[RaspberrySerialPort],
    queue: Queue,
    *args,
    **kwargs,
):
    while True:
        command_data = queue.get()
        temperature_data_logger.info(command_data)
        # 根据camera判断使用哪个串口
        if 'camera' in command_data:
            if len(serial_ports) == 2:
                ser = serial_ports[0] if command_data['camera'] == "0" else serial_ports[1]
            else:
                if command_data['camera'] == "1":
                    continue
                ser = serial_ports[0]
            command_message = ser.process_command(command_data)
            ser.message_sending(command_message)
        else:
            for ser in serial_ports:
                command_message = ser.process_command(command_data)
                ser.message_sending(command_message)