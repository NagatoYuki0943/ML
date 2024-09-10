from queue import Queue
import time
from algorithm import RaspberrySerialPort
from queue import Queue

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
        if 'camera' in command_data:
            ser = serial_ports[0] if command_data['camera'] == "1" else serial_ports[1]
            command_message = ser.process_command(command_data)
            ser.message_sending(command_message)
        else:
            for ser in serial_ports:
                command_message = ser.process_command(command_data)
                ser.message_sending(command_message)