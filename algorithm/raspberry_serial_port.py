import serial
import re
from datetime import datetime
import json


class RaspberrySerialPort:
    def __init__(
        self,
        temperature_logger,
        port: str = "/dev/ttyAMA2",
        baudrate: int = 115200,
        timeout: float = 0.0,
        BUFFER_SIZE: int = 2048,
    ):
        """
        Args:
            temperature_logger: 温度数据记录器.
            port (str): 端口号. Defaults to "/dev/ttyAMA2".
            baudrate (int): 波特率. Defaults to 115200.
            timeout (float): 阻塞时间. Defaults to 0.0.
            BUFFER_SIZE(int): 消息缓冲区大小. Defaults to 2048.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffer = ""
        self.BUFFER_SIZE = BUFFER_SIZE
        self.temperature_logger = temperature_logger

        # 预编译表达式
        self.cmd_pattern = re.compile(r'\$cmd')
        self.msgid_pattern = re.compile(r'&msgid=(\d+)')

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
            temperature_message = self.comm.read(self.comm.in_waiting).decode().strip()
            self.buffer += temperature_message
            self.temperature_logger.info(f"Received temperature message: {temperature_message}")

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
            if match_mid and match_mid.start() > match_end.start():
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
            self.temperature_logger.info(f"Processed temperature data: {temperature_data}")
            return temperature_data

    def extract_temperature_data(self, message):
        """解析串口消息, 组合成JSON格式

        Args:
            message (str): 一条完整的串口消息.
        """
        # 获取当前时间，表示此消息获取的时间
        time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        # 找出cmd参数所在位置
        cmd_start = message.find('cmd=') + len('cmd=')
        cmd_end = message.find('&param')

        # 找出param参数（温度数据）所在位置
        param_start = cmd_end + len('&param')
        param_end = message.find('&msgid=')
        if cmd_start == -1 or cmd_end == -1 or \
        param_start == -1 or param_end == -1:
            return None

        # 找出cmd、param、msgid的对应参数
        cmd = message[cmd_start:cmd_end]
        param_str = message[param_start:param_end].strip()
        msgid = message[param_end + len('&msgid='):]

        # contro_way转换成标准json格式
        if 'control_way' in param_str:
            param_str = param_str.replace('"control_way":warm', '"control_way":"warm"').\
                replace('"control_way":cold', '"control_way":"cold"')

        # 合并参数为字典
        data = {
            "cmd": cmd,
            "param": json.loads(param_str),
            "msgid": int(msgid),
            "timestamp": time
        }

        # 转为JSON串
        return data

    def message_sending(self, command_message):
        """发送串口消息

        Args:
            command_message (str): 控制指令消息.
        """
        self.comm.write(command_message.encode().strip())
        self.temperature_logger.info(f"Sending control message: {command_message}")

    def process_command(self, command_data):
        """合并控制指令与数据

        Args:
            command_data (dict): 控制指令与数据.
        """
        cmd = command_data['cmd']
        param = command_data['param']
        msgid = command_data['msgid']
        param_str = json.dumps(param, separators=(',', ':')).strip()
        command_message = f"$cmd={cmd}&param{param_str}&msgid={msgid}"
        return command_message
