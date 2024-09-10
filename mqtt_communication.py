import time
from algorithm import RaspberryMQTT
from queue import Queue
import json
import os
import subprocess
from config import MQTTConfig, RingsLocationConfig, CameraConfig, MatchTemplateConfig

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
        cmd_handlers = {
            'reboot': reboot_device,
            'setconfig': config_setter,
            'getconfig': config_getter
        }
        handler = cmd_handlers.get(message.get('cmd'))
        if handler:
            handler(message, send_queue)
        else:
            main_queue.put(message)

    # 检查设备是否由服务器命令重启
    if os.path.exists(MQTTConfig.getattr('staging_file')):
        file = handle_staging("load")
        reply = create_message(
            "reboot",
            "succ",
            {"code":200,"msg":"reboot succ"},
            file['msgid']
        )
        send_queue.put(reply)
        os.remove(MQTTConfig.getattr('staging_file'))

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
        print(topic + payload)
        client.publish(topic, payload)

def handle_staging(type, status = True, msgid = 1):
    """操作设备重启状态文件"""
    file = MQTTConfig.getattr('staging_file')
    if type == "load":
        with open(file, 'r') as f:
            return json.load(f)
    elif type == "save":
        with open(file, 'w') as f:
            json.dump({"reboot_pending":status, "msgid":msgid}, f)

def reboot_device(message, send_queue):
    """设备重启"""
    handle_staging("save", True, message['msgid'])
    try:
        subprocess.run(
            ['sudo', 'reboot', 'now'],
            check=True, 
            capture_output=True, 
            text=True
        )
    except subprocess.CalledProcessError as e:
        handle_config_exception(e, "reboot", message, send_queue)
        os.remove(MQTTConfig.getattr('staging_file'))

def config_setter(message, send_queue):
    """根据命令中携带的配置参数修改配置"""
    try:
        config_body = message.get("body", {})
        # 动态更新配置
        for key, value in config_body.items():
            if key not in config_map():
                handle_config_exception("not a valid config key", "setconfig", message, send_queue)
                return None
            config_class, config_attr = config_map()[key]
            config_class.setattr(config_attr, value)
            print(config_class.getattr(config_attr))
        reply = create_message(
            "setconfig", 
            "succ", 
            {"code":200,"msg":f"setconfig succeed"}, 
            message['msgid']
        )
        send_queue.put(reply)
    except Exception as e:
        handle_config_exception(e, "setconfig", message, send_queue)

def config_getter(message, send_queue):
    """获取所有的配置项"""
    try:
        # 存储配置项和值的字典
        config_body = {}
        # 遍历配置映射，获取每个配置项的当前值
        for key, config in config_map().items():
            if config is not None:
                config_class, config_attr = config
                config_body[key] = config_class.getattr(config_attr)
            else:
                config_body[key] = "not available"
        config_body['code'] = 200
        config_body['msg'] = "getconfig succeed" 
        # 返回成功的消息，包含所有配置项的当前值
        reply = create_message(
            "getconfig",
            "succ",
            config_body,
            message['msgid']
        )
        send_queue.put(reply)
    except Exception as e:
        # 返回失败的消息
        handle_config_exception(e, "getconfig", message, send_queue)

def handle_config_exception(e, cmd, message, send_queue):
    """报错响应模板"""
    reply = create_message(
        cmd,
        "fail",
        {"code":400,"msg":f"{cmd} failed: {str(e)}"},
        message['msgid']
    )
    send_queue.put(reply)

def create_message(cmd, result, body, msgid):
    """响应消息模板"""
    return {
        "cmd":cmd,
        "result":result,
        "body":body,
        "msgid":msgid
        }

def config_map():
    """关注的配置项"""
    return {
        'displacement_threshold': None,  # 尚未定义的配置
        'target_number': (MatchTemplateConfig, 'target_number'),
        'target_size': None,  # 尚未定义的配置
        'reference_target': None,  # 尚未定义的配置
        'data_report_interval': (CameraConfig, 'return_image_time_interval'),
        'capture_interval': (CameraConfig, 'capture_time_interval'),
        'device_sn': (MQTTConfig, 'device_sn'),
        'max_target_number': None,  # 尚未定义的配置
        'log_level': (MQTTConfig, 'log_level')
    }