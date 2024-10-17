import os
import time
from algorithm import RaspberryMQTT, RaspberryFTP
from queue import Queue
from config import MQTTConfig, RingsLocationConfig, CameraConfig, MatchTemplateConfig, MainConfig, FTPConfig
from loguru import logger
import sys

# MQTT 客户端接收线程
def mqtt_receive(
        client: RaspberryMQTT,
        main_queue: Queue,
        send_queue: Queue,
        *args,
        **kwargs,
):
    client.connect_mqtt()
    ftp = ftp_object_create()
    # 设置消息回调
    client.set_message_callback(
        lambda msg: message_handler(msg, send_queue, ftp, main_queue)
    )
    # 检查设备是否由命令重启
    if os.path.exists("reboot_info.txt"):
        with open("reboot_info.txt", "r") as f:
            msgid = f.read()
        create_message(
            "reboot",
            {"code":200,"msg":"reboot succeed","at":get_current_time(), "did":MQTTConfig.getattr('did')},
            msgid,
            send_queue
        )
        os.remove("reboot_info.txt")
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
        while not client.connection_flag:
            if queue.full():
                logger.warning(f"MQTT send queue is full, delete message {message}")
                message = queue.get()
            time.sleep(1)
        body = message.get("body", {})
        cmd = message.get("cmd")
        try:
            if "img" in body:
                timestamp = body['at'].replace("-", "").replace(":", "").replace(" ", "")
                ftpurl = f"{FTPConfig.getattr('image_base_url')}/{cmd}/{timestamp}"
                ftp.upload_file(body['path'], body['img'], ftpurl)
                message['body'].pop('path')
                message['body']['ftpurl'] = ftpurl
            elif "config" in body:
                timestamp = body['at'].replace(" ", "").replace("-", "").replace(":", "")
                ftpurl = f"{FTPConfig.getattr('config_base_url')}/{cmd}/{timestamp}"
                ftp.upload_file(body['path'], body['config'], ftpurl)
                message['body'].pop('path')
                message['body']['ftpurl'] = ftpurl
        except Exception as e:
            message['body']['code'] = 400
            message['body']['msg'] = f"{cmd} uploads faild: {e}"
            message['body'].pop('path')
            message['body']['ftpurl'] = ftpurl
        topic, payload = client.merge_message(message)
        client.publish(topic, payload)

def ftp_object_create():
    return RaspberryFTP(
        FTPConfig.getattr('ip'),
        FTPConfig.getattr('port'),
        FTPConfig.getattr('username'),
        FTPConfig.getattr('password'),
        FTPConfig.getattr('max_retries'),
        FTPConfig.getattr('delay'),
        FTPConfig.getattr('pasv_mode'),
    )

def message_handler(message, send_queue, ftp, main_queue):
    """MQTT消息处理"""
    cmd = message.get('cmd', 'unknown')
    cmd_handlers = {
        'setconfig': config_setter,
        'getconfig': config_getter,
        'updateconfigfile': config_file_update,
        'reboot': device_reboot,
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
    config_file_path = MainConfig.getattr("save_dir") / f"config_{timestamp}.yaml"
    try:
        ftp.download_file(config_file_path, body['config'], body['ftpurl'])
        message['body']['path'] = config_file_path
        message['body'].pop('ftpurl')
        return message
    except Exception as e:
        create_message(cmd, {"code":400,"msg":f"{cmd} failed:{e}",
        "at":get_current_time(), "did":MQTTConfig.getattr('did')}, msgid, send_queue)
        return None

def config_setter(message, send_queue):
    """根据命令中携带的配置参数修改配置"""
    cmd = "setconfig"
    msgid = message.get("msgid", "unknown")
    failed_keys = []
    map = config_map()
    # 根据key来选择配置的参数
    try:
        config_body = message.get("body", {})
        for key, value in config_body.items():
            # 检查配置项是否存在于设备中
            if key not in map:
                failed_keys.append(f"{key} is not a valid key")
                continue
            # 获取当前配置
            config_class = map[key]
            # 根据key做特殊处理
            if key == "camera0_reference_target_id2offset" or key == "camera1_reference_target_id2offset":
                if not isinstance(value, dict):
                    failed_keys.append(f"{key} must be a dictionary")
                # 使用迭代器直接获取第一个值
                target_key, target_val = next(iter(value.items()))
                if not isinstance(target_val, list) or len(target_val) != 2:
                    failed_keys.append(f"{key}'s value must be a list of 2 floats")
                    continue
                try:
                    # target_id转换为int类型
                    target_key_int = int(target_key)
                except ValueError:
                    failed_keys.append(f"{key} keys must be convertible to int")
                    continue
                value = [target_key_int, target_val]
                config_class.setattr(key, value)
                continue
            # 当前配置中的值
            current_value = config_class.getattr(key)
            # 处理配置中默认为tuple类型的参数
            if isinstance(current_value, tuple):
                if not isinstance(value, (list, tuple)) or len(value) != len(current_value):
                    failed_keys.append(f"{key} must be a tuple with {len(current_value)} values")
                    continue
                config_class.setattr(key, tuple(value))
                continue
            # 设置普通类型参数
            config_class.setattr(key, value)
        if not failed_keys:
            create_message(cmd, {"code":200,"msg":f"{cmd} succeed",
            "at":get_current_time(), "did":MQTTConfig.getattr('did')}, msgid, send_queue)
        else:
            create_message(cmd, {"code":206,"msg":f"{cmd} failed, failed keys:{failed_keys}",
            "at":get_current_time(), "did":MQTTConfig.getattr('did')}, msgid, send_queue)
    except Exception as e:
        create_message(cmd, {"code":400,"msg":f"{cmd} failed:{e}",
        "at":get_current_time(), "did":MQTTConfig.getattr('did')}, msgid, send_queue)

def config_getter(message, send_queue):
    """获取所有的配置项"""
    cmd = "getconfig"
    msgid = message.get("msgid", "unknown")
    try:
        # 存储配置项和值
        config_body = {}
        # 遍历配置映射，获取每个配置项的当前值
        for key, config_class in config_map().items():
            value = config_class.getattr(key)
            if isinstance(value, tuple):
                config_body[key] = list(value)
            else:
                config_body[key] = value
        config_body['code'] = 200
        config_body['msg'] = "getconfig succeed"
        config_body['at'] = get_current_time()
        config_body['did'] = MQTTConfig.getattr('did') 
        # 返回成功的消息，包含所有配置项的当前值
        create_message(cmd, config_body, msgid, send_queue)
    except Exception as e:
        # 返回失败的消息
        create_message(cmd, {"code":400,"msg":f"{cmd} failed:{e}",
        "at":get_current_time(), "did":MQTTConfig.getattr('did')}, msgid, send_queue)

def device_reboot(message, send_queue):
    msgid = message['msgid']
    cmd = "reboot"
    type = message['type']
    if os.path.exists("reboot_info.txt"):
        return
    try:
        with open("reboot_info.txt", "w") as f:
            f.write(msgid)
        if type == "device":
            logger.info("Device will be reboot in 1 second")
            time.sleep(1)
            os.system("sudo reboot")
        elif type == "process":
            logger.info("Process will be reboot in 1 second")
            time.sleep(1)
            python = sys.executable
            os.execv(python, [python] + sys.argv)
    except Exception as e:
        logger.error(f"Device reboot faild: {e}")
        create_message(cmd, {"code":400,"msg":f"{cmd} faild:{e}",
        "at":get_current_time(), "did":MQTTConfig.getattr('did')},msgid, send_queue)



def create_message(cmd, body, msgid, send_queue):
    """响应消息模板"""
    body['did'] = MQTTConfig.getattr("did")
    reply = {
        "cmd":cmd,
        "body":body,
        "msgid":msgid
        }
    send_queue.put(reply)

def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def config_map():
    """设备中需要修改或查询的配置项"""
    return {
        'x_move_threshold': RingsLocationConfig,  # X轴位移告警阈值
        'y_move_threshold': RingsLocationConfig,  # X轴位移告警阈值
        'z_move_threshold': RingsLocationConfig,  # X轴位移告警阈值
        'target_number': MatchTemplateConfig, # 靶标数量
        'template_size': MatchTemplateConfig,  # 靶标尺寸
        'camera0_reference_target_id2offset': RingsLocationConfig, # 参考靶标
        'camera1_reference_target_id2offset': RingsLocationConfig, # 参考靶标
        'cycle_time_interval': MainConfig, # 上报数据时间间隔
        'capture_time_interval': CameraConfig, # 拍照时间间隔
        'did': MQTTConfig, # divece id
        'max_target_number': MatchTemplateConfig, # 最大支持靶标个数
        'log_level': MainConfig, # 日志等级
    }
