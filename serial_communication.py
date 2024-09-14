from queue import Queue
import time
from algorithm import RaspberrySerialPort
from queue import Queue
from datetime import datetime
from loguru import logger

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
                logger.info(f"Received serial port message: {temperature_data}")
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
        # 根据camera判断使用哪个串口
        if 'camera' in command_data:
            ser = serial_ports[0] if command_data['camera'] == "1" else serial_ports[1]
            command_message = ser.process_command(command_data)
            logger.info(f"Send serial port message: {command_message}")
            ser.message_sending(command_message)
        else:
            for ser in serial_ports:
                command_message = ser.process_command(command_data)
                logger.info(f"Send serial port message: {command_message}")
                ser.message_sending(command_message)

def serial_for_test(
    queue: Queue,
    main_queue: Queue,
    *args,
    **kwargs,
):
    mode = 1
    i = 1
    while True:
        response = None
        if not queue.empty():
            command_data = queue.get()
            logger.info(f"Send serial message: {command_data}")
            cmd = command_data['cmd']
            if cmd == "adjusttempdata":
                mode = 2
                response = {
                    "cmd":"askadjusttempdata",
                    "times":datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                    "camera":"1",
                    "param":{
                        "result":"OK"
                    },
                    "msgid":command_data['msgid']
                }
            if cmd == "adjustLEDlevel":
                response = {
                    "cmd":"askadjustLEDlevel",
                    "times":datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                    "param":{
                        "result":"OK"
                    },
                    "msgid":command_data['msgid']
                }
            if response:
                logger.info(f"Received serial port message: {response}")
                main_queue.put(response)
                continue
        if mode == 1:
            response = {
                "cmd":"sendtempdata",
                "camera":"1",
                "times":datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                "param":{
                    "inside_air_t":10,
                    "exterior_air_t":10,
                    "sensor1_t":10,
                    "sensor2_t":10,
                    "sensor3_t":10,
                    "sensor4_t":257,
                    "sensor5_t":257,
                    "sensor6_t":257
                },
                "msgid":i
            }
        elif mode == 2:
            response = {
                "cmd":"sendadjusttempdata",
                "camera":"1",
                "times":datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                "param":{
                    "parctical_t":10,
                    "control_t":10,
                    "control_way":"warm",
                    "pwm_data":10
                },
                "msgid":i
            }
        if response:
            logger.info(f"Received serial port message: {response}")
            main_queue.put(response)
        i = 1 if i > 100 else i + 1
        time.sleep(10)
