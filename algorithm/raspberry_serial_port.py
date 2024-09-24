import serial
import re
from datetime import datetime
import json
from threading import Lock
import os
from loguru import logger


class RaspberrySerialPort:
    def __init__(
        self,
        temperature_data_save_path: str = "/log/temperature_data.json",
        port: str = "/dev/ttyAMA2",
        baudrate: int = 115200,
        timeout: float = 0.0,
        BUFFER_SIZE: int = 2048,
        LOG_SIZE: int = 1_000_000,
    ):
        """
        Args:
            temperature_data_save_path: 温度数据记录文件路径. Defaults to "/log/temperature_data.json".
            port (str): 端口号. Defaults to "/dev/ttyAMA2".
            baudrate (int): 波特率. Defaults to 115200.
            timeout (float): 阻塞时间. Defaults to 0.0.
            BUFFER_SIZE(int): 消息缓冲区大小. Defaults to 2KB.
            LOG_SIZE(int): 温度数据日志大小. Defaults to 1MB.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffer = ""
        self.BUFFER_SIZE = BUFFER_SIZE
        self.temperature_data_save_path = temperature_data_save_path
        self.LOG_SIZE = LOG_SIZE
        self.lock = Lock()

        # 预编译表达式
        self.cmd_pattern = re.compile(r'\$cmd=([^&]+)')
        self.msgid_pattern = re.compile(r'&msgid=(\d+)')
        self.param_pattern = re.compile(r'&param({.*})')

        # 开启串口
        self.comm = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout
        )

    def close(self):
        """关闭串口"""
        self.comm.close()

    def message_reception(self):
        """获取串口消息, 放入缓冲区"""
        if self.comm.in_waiting > 0:
            with self.lock:
                temperature_message = self.comm.read(self.comm.in_waiting).decode().strip()
                self.buffer += temperature_message
            logger.info(f"Received serial message from serial port: {self.port}, message: {temperature_message}")

    def subcontracting(self):
        """处理串口数据包"""
        # 处理缓冲区超过大小的情况，替换为最新数据
        if len(self.buffer) > self.BUFFER_SIZE:
            self.buffer = self.buffer[-self.BUFFER_SIZE:]

        # 查找包头和包尾
        match_head = self.cmd_pattern.search(self.buffer)
        match_end = self.msgid_pattern.search(self.buffer)

        # 抛弃不理想的数据包（包尾在前），保留不完整的包（包头在前但没有包尾）
        if not match_head:
            self.buffer = ""
            return None
        elif not match_end:
            return None
        elif match_head.start() > match_end.end():
            self.buffer = self.buffer[match_head.start():]
            return None
        else:
            # 确保有包头包尾的情况下再检查是否多个包粘连
            match_mid = self.cmd_pattern.search(self.buffer, match_head.end())
            if match_mid and match_mid.start() < match_end.start():
                self.buffer = self.buffer[match_mid.start():]
                return None

            # 一条完整包的处理
            start_idx = match_head.start()
            end_idx = match_end.end()
            complete_message = self.buffer[start_idx:end_idx]
            # 将消息解析
            temperature_data = self.extract_temperature_data(complete_message)
            # 更新缓冲区
            self.buffer = self.buffer[end_idx:]
            return temperature_data

    def extract_temperature_data(self, message):
        """解析串口消息, 组合成JSON格式

        Args:
            message (str): 一条完整的串口消息.
        """
        # 获取当前时间，表示此消息获取的时间
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        result = {}

        cmd_match = self.cmd_pattern.search(message)
        if cmd_match:
            result['cmd'] = cmd_match.group(1)
        
        param_match = self.param_pattern.search(message)
        if param_match:
            param_json = param_match.group(1)
            try:
                result['param'] = json.loads(param_json)
            except json.JSONDecodeError:
                result['param'] = param_json
                
        msgid_match = self.msgid_pattern.search(message)
        if msgid_match:
            result['msgid'] = msgid_match.group(1)

        # 合并参数为字典
        result['times'] = time
        camera = "1" if self.port == "/dev/ttyAMA1" else "2"
        result['camera'] = camera

        # 转为JSON串
        return result

    def message_sending(self, command_message):
        """发送串口消息

        Args:
            command_message (str): 控制指令消息.
        """
        with self.lock:
            self.comm.write(command_message.encode().strip())
        logger.info(f"Sending message to serial port: {self.port}, message: {command_message}")

    def process_command(self, command_data):
        """合并控制指令与数据

        Args:
            command_data (dict): 控制指令与数据.
        """
        cmd = command_data.get('cmd', "unknown")
        param = command_data.get('param', {})
        msgid = command_data.get('msgid', "unknown")
        param_str = json.dumps(param, separators=(',', ':')).strip()
        command_message = f"$cmd={cmd}&param{param_str}&msgid={msgid}"
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        command_data['times'] = time
        self.save_temperature_data(command_data)
        return command_message
    
    def save_temperature_data(self, data):
        """记录温度数据
        
        Args:
            data (dict): 需要记录的数据
        """
        self.check_file_size()
        with open(self.temperature_data_save_path, "a") as file:
            json.dump(data, file)
            file.write("\n")
    
    def check_file_size(self):
        """检查温度数据记录文件大小"""
        if os.path.exists(self.temperature_data_save_path):
            if os.path.getsize(self.temperature_data_save_path) > self.LOG_SIZE:
                with open(self.temperature_data_save_path, "w") as file:
                    file.truncate(0)
            else:
                return