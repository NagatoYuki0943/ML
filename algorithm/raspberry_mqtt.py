from paho.mqtt import client as mqtt
from loguru import logger


class RaspberryMQTT:
    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        timeout: int = 60,
        topic: str = "test/topic",
    ):
        """
        Args:
            mqtt_logger: MQTT消息记录器.
            broker (str): 服务器. Defaults to "localhost".
            port (int): 端口号. Defaults to 1883
            timeout (float): 连接超时时间. Defaults to 60
            topic (str): 订阅的主题. Defaults to "test/topic"
        """
        self.broker = broker
        self.port = port
        self.timeout = timeout
        self.topic = topic

        self.client = mqtt.Client()
        self.message_callback = None
        self.client.on_message = self.on_message
        self.connect_mqtt()

    def connect_mqtt(self):
        """与服务器建立连接并订阅主题"""
        def on_connect(client, userdata, flags, rc):
            logger.info(f"MQTT connection status:{rc}")
        self.client.on_connect = on_connect
        self.client.connect(host = self.broker, port = self.port, keepalive = self.timeout)
        self.client.subscribe(self.topic)

    def on_message(self, client, userdata, msg):
        """收到的消息传入回调函数"""
        message = msg.payload.decode('utf-8')
        if self.message_callback:
            self.message_callback(message)

    def set_message_callback(self, callback):
        """设置回调函数"""
        self.message_callback = callback

    def publish(self, payload):
        """发送消息"""
        self.client.publish(self.topic, payload)

    def loop(self):
        """事件循环"""
        self.client.loop()

    def stop(self):
        """断开服务器连接"""
        self.client.disconnect()
