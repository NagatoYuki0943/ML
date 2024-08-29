import time
from algorithm import RaspberryMQTT
from queue import Queue


# MQTT 客户端接收线程
def mqtt_receive(
        client: RaspberryMQTT,
        queue: Queue,
        *args,
        **kwargs,
):
    # 消息处理-放入主控消息队列
    def message_handler(message):
        queue.put(message)
    client.set_message_callback(message_handler)
    while True:
        client.loop()
        time.sleep(0.1)


# MQTT客户端发送线程
def mqtt_send(
        client: RaspberryMQTT,
        queue: Queue,
        *args,
        **kwargs,
):
    while True:
        message = queue.get()
        client.publish(message)