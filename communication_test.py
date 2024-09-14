import time
from threading import Thread
from queue import Queue
from loguru import logger
import json
import os

from algorithm import (
    ThreadWrapper,
    RaspberryMQTT,
)
from config import (
    MQTTConfig,
)
from mqtt_communication import mqtt_receive, mqtt_send
from serial_communication import serial_receive, serial_send, serial_for_test

flag = False

def receive(queue):
    global flag
    while flag:
        if not queue.empty():
            print(queue.get_nowait())
        time.sleep(0.1)

def main():

    main_queue = Queue()
#----------初始化MQTT客户端----------#
    mqtt_comm = RaspberryMQTT(
        MQTTConfig.getattr('broker'),
        MQTTConfig.getattr('port'),
        MQTTConfig.getattr('timeout'),
        MQTTConfig.getattr('topic'),
        MQTTConfig.getattr('username'),
        MQTTConfig.getattr('password'),
        MQTTConfig.getattr('clientId'),
        apikey=MQTTConfig.getattr('apikey')
    )
    mqtt_send_thread = ThreadWrapper(
        target_func = mqtt_send,
        client = mqtt_comm,
    )
    mqtt_send_queue = mqtt_send_thread.queue
    mqtt_receive_thread = Thread(
        target = mqtt_receive,
        kwargs={'client':mqtt_comm, 'main_queue':main_queue, 'send_queue':mqtt_send_queue},
    )
    serial_test_thread = ThreadWrapper(
        target_func = serial_for_test,
        main_queue = main_queue
    )
    # serial_test_thread.start()
    mqtt_receive_thread.start()
    mqtt_send_thread.start()
    logger.success("初始化MQTT客户端完成")
    #----------初始化MQTT客户端----------#

    with open('mqtt_message.json', 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    
    receive_thread = Thread(target=receive , args=(main_queue,))

    try:
        global flag
        flag = True
        receive_thread.start()
        while flag:
            cmd = input("input command's titles")
            if cmd in config:
                mqtt_send_queue.put(config[cmd])
            else:
                print("no such command")
            time.sleep(1)
    except KeyboardInterrupt:
        flag=False
        os._exit()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
