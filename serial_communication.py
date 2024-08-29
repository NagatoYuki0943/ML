from queue import Queue
import time
from config import SerialCommConfig
from algorithm import RaspberrySerialPort
from queue import Queue

# 接收线程
def serial_receive(
        ser: RaspberrySerialPort,
        queue: Queue,
        *args,
        **kwargs
):
    while True:
        with SerialCommConfig.lock:
            # 接收消息
            ser.message_reception()
            # 处理缓冲区中的消息并解析
            temperature_data = ser.subcontracting()
        if temperature_data is not None:
            queue.put(temperature_data)
        time.sleep(0.1)

# 发送线程
def serial_send(
    ser: RaspberrySerialPort,
    queue: Queue,
    *args,
    **kwargs,
):
    while True:
        command_data = queue.get()
        # 格式转换
        command_message = ser.process_command(command_data)
        with SerialCommConfig.lock:
            ser.message_sending(command_message)