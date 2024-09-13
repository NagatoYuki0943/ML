import time
from algorithm import RaspberryMQTT
from queue import Queue
from config import MQTTConfig, RingsLocationConfig, CameraConfig, MatchTemplateConfig
from loguru import logger

# MQTT 客户端接收线程
def mqtt_receive(
        client: RaspberryMQTT,
        main_queue: Queue,
        send_queue: Queue,
        *args,
        **kwargs,
):
    def message_handler(message):
        """MQTT客户端消息回调"""
        logger.info(f"Received MQTT message: {message}")
        cmd_handlers = {
            'setconfig': config_setter,
            'getconfig': config_getter
        }
        # 根据cmd判断消息是否发给主控处理
        handler = cmd_handlers.get(message.get('cmd'))
        if handler:
            handler(message, send_queue)
        else:
            main_queue.put(message)
    # 设置消息回调
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
        topic, payload = client.merge_message(message)
        logger.info(f"Send MQTT message: {payload} in topic: {topic}")
        client.publish(topic, payload)


def config_setter(message, send_queue):
    """根据命令中携带的配置参数修改配置"""
    cmd = "setconfig"
    msgid = message.get("msgid", "unknown")
    # 根据key来选择配置的参数
    try:
        config_body = message.get("body", {})
        for key, value in config_body.items():
            # 检查配置项是否存在于设备中
            if key not in config_map():
                raise ValueError(f"{key} is not a valid key")
            config_class, config_attr = config_map()[key]
            config_class.setattr(config_attr, value)
        create_message(cmd, {"code":200,"msg":f"{cmd} succeed"}, msgid, send_queue)
    except Exception as e:
        create_message(cmd, {"code":400,"msg":f"{cmd} failed:{e}"}, msgid, send_queue)

def config_getter(message, send_queue):
    """获取所有的配置项"""
    cmd = "getconfig"
    msgid = message.get("msgid", "unknown")
    try:
        # 存储配置项和值
        config_body = {}
        # 遍历配置映射，获取每个配置项的当前值
        for key, config in config_map().items():
            config_class, config_attr = config
            config_body[key] = config_class.getattr(config_attr)
        config_body['code'] = 200
        config_body['msg'] = "getconfig succeed" 
        # 返回成功的消息，包含所有配置项的当前值
        create_message(cmd, config_body, msgid, send_queue)
    except Exception as e:
        # 返回失败的消息
        create_message(cmd, {"code":400,"msg":f"{cmd} failed:{e}"}, msgid, send_queue)

def create_message(cmd, body, msgid, send_queue):
    """响应消息模板"""
    body['did'] = MQTTConfig.getattr("did")
    reply = {
        "cmd":cmd,
        "body":body,
        "msgid":msgid
        }
    send_queue.put(reply)

def config_map():
    """设备中需要修改或查询的配置项"""
    return {
        'displacement_threshold': "None",  # 尚未定义的配置
        'target_number': (MatchTemplateConfig, 'target_number'),
        'target_size': "None",  # 尚未定义的配置
        'reference_target': "None",  # 尚未定义的配置
        'data_report_interval': (CameraConfig, 'return_image_time_interval'),
        'capture_interval': (CameraConfig, 'capture_time_interval'),
        'did': (MQTTConfig, 'did'),
        'max_target_number': "None",  # 尚未定义的配置
        'log_level': (MQTTConfig, 'log_level')
    }