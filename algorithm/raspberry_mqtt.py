from paho.mqtt import client as mqtt
from loguru import logger
import json
import re

class RaspberryMQTT:
    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        timeout: int = 60,
        topic: str = "test",
        username: str = "admin",
        password: str = "123456",
        clientId: str = "123456",
        apikey: str = "123456"
    ):
        """
        Args:
            mqtt_logger: MQTT消息记录器.
            broker (str): 服务器. Defaults to "localhost".
            port (int): 端口号. Defaults to 1883.
            timeout (float): 连接超时时间. Defaults to 60.
            topic (str): 订阅的主题. Defaults to "test/topic".
            username (str): 连接使用的用户名. Defaults to admin.
            password (str): 连接使用的密码.
            clientId (str): 使用设备SN码. Defaults to 7804d2.
            apikey (str): 鉴权码.
        """
        self.broker = broker
        self.port = port
        self.timeout = timeout
        self.topic = topic
        self.apikey = apikey

        self.client = mqtt.Client(client_id=clientId, clean_session=False)
        self.client.username_pw_set(username=username, password=password)
        self.message_callback = None
        self.client.on_message = self.on_message
        self.client.on_log = self.on_log
        self.connect_mqtt()
        self.pattern = re.compile(r'(\w+)=([^&]*)')

    def connect_mqtt(self):
        """与服务器建立连接并订阅主题"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logger.info(f"MQTT server connected")
            else:
                logger.error(f"MQTT server connection error")
        self.client.on_connect = on_connect
        self.client.connect(host = self.broker, port = self.port, keepalive = self.timeout )
        self.client.subscribe(self.topic, qos = 1)

    def on_message(self, client, userdata, msg):
        """收到的消息传入回调函数"""
        message = msg.payload.decode('utf-8')
        logger.info(f"Received message from server: {message}")
        data = self.extract_message(message)
        if data is not None:
            if self.message_callback:
                self.message_callback(data)

    def set_message_callback(self, callback):
        """设置回调函数"""
        self.message_callback = callback

    def publish(self, topic, payload):
        """发送消息"""
        logger.info(f"Sent message to server: {payload} in topic {topic}")
        self.client.publish(topic, payload)

    def loop(self):
        """事件循环"""
        self.client.loop()

    def stop(self):
        """断开服务器连接"""
        self.client.disconnect()

    def extract_message(self, message):
        """解析mqtt消息

        Args:
            message (str): 接收的数据.
        """
        try:
            matches = self.pattern.findall(message)
            data = {key: value for key, value in matches}
            if data['apikey'] != self.apikey:
                return None
            del data['apikey']
            if 'body' in data:
                body_content = data['body']
                try:
                    body_dict = json.loads(body_content)
                    data['body'] = body_dict
                except json.JSONDecodeError as e:
                    print(f"Error parsing body JSON: {e}")
                    data['body'] = body_content
            if "cmd" not in data:
                return None
            return data
        except Exception as e:
            print(f"error extracting message:{e}")
            return None
    
    def merge_message(self, data):
        """线程消息组合为MQTT消息
        
        Args:
            data (dict): 要发送的数据.
        """
        cmd = data.get("cmd")
        if cmd == "update":
            topic = "$dp"
            did = data.get("did", "unknown")
            sensor_data = data.get("data", {})
            data_json = json.dumps(sensor_data, separators=(',',':')).strip()
            message = f'{{"{did}":{data_json}}}'.strip()
            return topic, message
        
        if cmd == "getstatus":
            topic = "$dr"
            status = data.get("body", {})
            status_json = json.dumps(status, separators=(",",":")).strip()
            msgid = data.get("msgid", "unknown")
            message = f"$cmd={cmd}&status={status_json}&msgid={msgid}"
            return topic, message
        
        topic = "$spepc"
        body = data.get("body", {})
        body_json = json.dumps(body, separators=(",",":")).strip()
        # 响应消息
        if cmd != "alarm" and cmd != "devicestate":
            msgid = data.get("msgid", "unknown")
            result = data.get("result", "unknown")
            message = f"$cmd={cmd}&result={result}&body={body_json}&msgid={msgid}"
        # 状态上报和告警消息
        else:
            message = f"$cmd={cmd}&body={body_json}".strip()
        return topic, message
