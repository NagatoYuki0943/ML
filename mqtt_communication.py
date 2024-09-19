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
        main_queue: Queue,
        send_queue: Queue,
        *args,
        **kwargs,
):
    ftp = ftp_object_create()
    # 设置消息回调
    client.set_message_callback(
        lambda msg: message_handler(msg, send_queue, ftp, main_queue)
    )
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
    ftp = ftp_object_create()
    while True:
        message = queue.get()
        body = message.get("body", {})
        cmd = message.get("cmd")
        if body:
            if "img" in body:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                ftpurl = f"{FTPConfig.getattr('image_base_url')}/{cmd}/{timestamp}"
                ftp.ftp_connect()
                ftp.upload_file(body['path'], body['img'], ftpurl)
                ftp.ftp_close()
                message['body'].pop('path')
                message['body']['ftpurl'] = ftpurl
            elif "config" in body:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                ftpurl = f"{FTPConfig.getattr('config_base_url')}/{cmd}/{timestamp}"
                ftp.ftp_connect()
                ftp.upload_file(body['path'], body['config'], ftpurl)
                ftp.ftp_close()
                message['body'].pop('path')
                message['body']['ftpurl'] = ftpurl
        topic, payload = client.merge_message(message)
        client.publish(topic, payload)

def ftp_object_create():
    return RaspberryFTP(
        FTPConfig.getattr('ip'),
        FTPConfig.getattr('port'),
        FTPConfig.getattr('username'),
        FTPConfig.getattr('password')
    )

def message_handler(message, send_queue, ftp, main_queue):
    """MQTT消息处理"""
    cmd = message.get('cmd', 'unknown')
    cmd_handlers = {
        'setconfig': config_setter,
        'getconfig': config_getter,
        'updateconfigfile': config_file_update,
    }
    handler = cmd_handlers.get(cmd)
    if handler:
        if cmd == 'updateconfigfile':
            update_message = handler(message, send_queue, ftp)
            if update_message is not None:
                main_queue.put(update_message)
        else:
            handler(message, send_queue)
    else:
        main_queue.put(message)

def config_file_update(message, send_queue, ftp: RaspberryFTP):
    """更新配置文件"""
    cmd = "updateconfigfile"
    msgid = message['msgid']
    body = message.get("body")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config_file_path = MainConfig.getattr("save_dir") / f"Configuration_{timestamp}.yaml"
    try:
        ftp.ftp_connect()
        ftp.download_file(config_file_path, body['config'], body['ftpurl'])
        ftp.ftp_close()
        message['body']['path'] = config_file_path
        message['body'].pop('ftpurl')
        return message
    except Exception as e:
        create_message(cmd, {"code":400,"msg":f"{cmd} failed:{e}"}, msgid, send_queue)
        return None

def config_setter(message, send_queue):
    """根据命令中携带的配置参数修改配置"""
    cmd = "setconfig"
    msgid = message.get("msgid", "unknown")
    map = config_map()
    success = True
    failed_keys = []
    # 根据key来选择配置的参数
    try:
        config_body = message.get("body", {})
        for key, value in config_body.items():
            # 检查配置项是否存在于设备中
            if key not in map:
                failed_keys.append(f"{key} is not a valid key")
                success = False
                continue
            config = map[key]
            config_class, config_attr = config
            current_value = getattr(config_class, config_attr)
            if isinstance(current_value, tuple):
                if not isinstance(value, (list, tuple)) or len(value) != len(current_value):
                    failed_keys.append(f"{key} must be a tuple with {len(current_value)} values")
                    success = False
                    continue
                setattr(config_class, config_attr, tuple(value))
                continue
            setattr(config_class, config_attr, value)
        if success:
            create_message(cmd, {"code":200,"msg":f"{cmd} succeed"}, msgid, send_queue)
        else:
            create_message(cmd, {"code":206,"msg":f"{cmd} partially failed, failed keys:{failed_keys}"}, msgid, send_queue)
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
            value = getattr(config_class, config_attr)
            if isinstance(value, tuple):
                config_body[key] = list(value)
            else:
                config_body[key] = value
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
