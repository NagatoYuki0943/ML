import os
import time
from algorithm import RaspberryMQTT, RaspberryFTP
from queue import Queue
from config import MQTTConfig, RingsLocationConfig, CameraConfig, MatchTemplateConfig, MainConfig, FTPConfig
from loguru import logger
import subprocess

# MQTT 客户端接收线程
def mqtt_receive(
        client: RaspberryMQTT,
        ftp: RaspberryFTP,
        main_queue: Queue,
        send_queue: Queue,
        *args,
        **kwargs,
):
    def message_handler(message):
        """MQTT客户端消息回调"""
        cmd = message.get('cmd', "unknown")
        cmd_handlers = {
            'setconfig': config_setter,
            'getconfig': config_getter,
        }
        # 根据cmd判断消息是否发给主控处理
        handler = cmd_handlers.get(cmd)
        if handler:
            handler(message, send_queue)
        else:
            # 处理配置文件下载命令
            if cmd == "updateconfigfile":
                msgid = message['msgid']
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                config_file_path = MainConfig.getattr("save_dir") / f"Configuration_{timestamp}.yaml"
                try:
                    ftp.download_file(config_file_path, message['ftpurl'])
                    message['configuration_path'] = config_file_path
                    message.pop('ftpurl')
                except Exception as e:
                    create_message(cmd, {"code":400,"msg":f"{cmd} failed:{e}"}, msgid, send_queue)
                    return
            main_queue.put(message)
    # 设置消息回调
    client.set_message_callback(message_handler)
    while True:
        client.loop()
        time.sleep(0.1)

# MQTT客户端发送线程
def mqtt_send(
        client: RaspberryMQTT,
        ftp: RaspberryFTP,
        queue: Queue,
        *args,
        **kwargs,
):
    while True:
        message = queue.get()
        body = message.get("body")
        if body and "img" in body:
            ftp.upload_file(body['img'], body['ftpurl'])
        topic, payload = client.merge_message(message)
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
            config = config_map()[key]
            if isinstance(config, tuple):
                config_class, config_attr = config
                current_value = getattr(config_class, config_attr)
                if isinstance(current_value, tuple):
                    if not isinstance(value, (list, tuple)) or len(value) != len(current_value):
                        raise ValueError(f"{key} must be a tuple with {len(current_value)} values")
                    setattr(config_class, config_attr, tuple(value))
                else:
                    setattr(config_class, config_attr, value)
            else:
                raise ValueError(f"{key} is not avaliable")
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
            if isinstance(config, tuple):
                config_class, config_attr = config
                value = getattr(config_class, config_attr)
                if isinstance(value, tuple):
                    config_body[key] = list(value)
                else:
                    config_body[key] = value
            else:
                config_body[key] = config
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
        'displacement_threshold': (RingsLocationConfig, 'move_threshold'),  # 告警位移阈值
        'target_number': (MatchTemplateConfig, 'target_number'), # 靶标数量
        'target_size': (MatchTemplateConfig, 'template_size'),  # 靶标尺寸
        'reference_target': (MatchTemplateConfig, 'reference_target_ids'), # 参考靶标
        'data_report_interval': (MainConfig, 'cycle_time_interval'), # 上报数据时间间隔
        'capture_interval': (CameraConfig, 'capture_time_interval'), # 拍照时间间隔
        'did': (MQTTConfig, 'did'), # divece id
        'max_target_number': (MatchTemplateConfig, 'max_target_number'),  # 最大支持靶标个数
        'log_level': (MainConfig, 'log_level') # 日志等级
    }
