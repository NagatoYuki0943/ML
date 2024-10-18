from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from threading import Thread
import queue
from loguru import logger
from pathlib import Path
import os

from algorithm import (
    ThreadWrapper,
    DualStereoCalibration,
    RaspberryMQTT,
    RaspberrySerialPort,
    sort_boxes_center,
    pixel_num2object_size,
)
from config import (
    MainConfig,
    DualStereoCalibrationConfig,
    MatchTemplateConfig,
    CameraConfig,
    RingsLocationConfig,
    AdjustCameraConfig,
    TemperatureConfig,
    MQTTConfig,
    SerialCommConfig,
    ALL_CONFIGS,
    load_stereo_calibration_config,
    init_config_from_yaml,
    load_config_from_yaml,
    save_config_to_yaml,
)
from camera_engine import camera_engine
from find_target import find_target, find_around_target, find_lost_target
from adjust_camera import (
    adjust_exposure_full_res,
    adjust_exposure_low_res,  # 调整分辨率需要一段时间才能获取调整后的图片分辨率
)
from location_utils import rings_location, init_standard_results, calc_move_distance
from serial_communication import serial_receive, serial_send
from mqtt_communication import mqtt_receive, mqtt_send
from utils import (
    drop_excessive_queue_items,
    save_to_jsonl,
    get_now_time,
    save_image,
    get_picture_timeout_process,
)
from fake_queue import FakeQueue


# 将日志输出到文件
# 每天 0 点新创建一个 log 文件
handler_id = logger.add(
    str(MainConfig.getattr("loguru_log_path")),
    level=MainConfig.getattr("log_level"),
    rotation="00:00",
)


# ------------------------------ 初始化 ------------------------------ #
logger.info("init start")

# -------------------- 载入畸变矫正 -------------------- #
load_stereo_calibration_config()
# -------------------- 载入畸变矫正 -------------------- #

# -------------------- 基础 -------------------- #
# 主线程消息队列
main_queue = queue.Queue()
image0_timestamp: str
image0: np.ndarray
image0_metadata: dict
# -------------------- 基础 -------------------- #

# -------------------- 初始化相机 -------------------- #
logger.info("开始初始化相机")
camera0_thread = ThreadWrapper(
    target_func=camera_engine,
    queue_maxsize=CameraConfig.getattr("queue_maxsize"),
    camera_index=0,
)
camera0_thread.start()
camera0_queue = camera0_thread.queue


camera1_thread = ThreadWrapper(
    target_func=camera_engine,
    queue_maxsize=CameraConfig.getattr("queue_maxsize"),
    camera_index=1,
)
camera1_thread.start()
camera1_queue = camera1_thread.queue
time.sleep(1)
logger.success("初始化相机完成")
# -------------------- 初始化相机 -------------------- #

# -------------------- 畸变矫正 -------------------- #
logger.info("开始初始化畸变矫正")
dual_stereo_calibration = DualStereoCalibration(
    DualStereoCalibrationConfig.getattr("camera_matrix_left"),
    DualStereoCalibrationConfig.getattr("camera_matrix_right"),
    DualStereoCalibrationConfig.getattr("distortion_coefficients_left"),
    DualStereoCalibrationConfig.getattr("distortion_coefficients_right"),
    DualStereoCalibrationConfig.getattr("R"),
    DualStereoCalibrationConfig.getattr("T"),
    DualStereoCalibrationConfig.getattr("pixel_width_mm"),
)
logger.success("初始化畸变矫正完成")
# -------------------- 畸变矫正 -------------------- #

# -------------------- 初始化串口 -------------------- #
logger.info("开始初始化串口")
try:
    serial_objects = []

    for port in [
        SerialCommConfig.getattr("camera0_ser_port"),
        SerialCommConfig.getattr("camera1_ser_port"),
    ]:
        if port:
            object = RaspberrySerialPort(
                port,
                SerialCommConfig.getattr("baudrate"),
                SerialCommConfig.getattr("timeout"),
                SerialCommConfig.getattr("BUFFER_SIZE"),
            )
            serial_objects.append(object)

    serial_send_thread = ThreadWrapper(
        target_func=serial_send,
        serial_ports=serial_objects,
    )
    serial_receive_thread = Thread(
        target=serial_receive,
        kwargs={
            "serial_ports": serial_objects,
            "queue": main_queue,
        },
    )
    serial_send_queue = serial_send_thread.queue
    serial_receive_thread.start()
    serial_send_thread.start()
    logger.success("初始化串口完成")
except Exception as e:
    serial_send_queue = FakeQueue()
    logger.error(f"初始化串口失败: {e}, use fake queue instead")
# -------------------- 初始化串口 -------------------- #

# -------------------- 初始化MQTT客户端 -------------------- #
logger.info("开始初始化MQTT客户端")
use_mqtt = False
if use_mqtt:
    mqtt_comm = RaspberryMQTT(
        MQTTConfig.getattr("broker"),
        MQTTConfig.getattr("port"),
        MQTTConfig.getattr("timeout"),
        MQTTConfig.getattr("topic"),
        MQTTConfig.getattr("username"),
        MQTTConfig.getattr("password"),
        MQTTConfig.getattr("clientId"),
        MQTTConfig.getattr("apikey"),
    )
    mqtt_send_thread = ThreadWrapper(
        target_func=mqtt_send,
        queue_maxsize=MQTTConfig.getattr("send_queue_maxsize"),
        client=mqtt_comm,
    )
    mqtt_send_queue = mqtt_send_thread.queue
    mqtt_receive_thread = Thread(
        target=mqtt_receive,
        kwargs={
            "client": mqtt_comm,
            "main_queue": main_queue,
            "send_queue": mqtt_send_queue,
        },
    )
    mqtt_receive_thread.start()
    mqtt_send_thread.start()
    logger.success("初始化MQTT客户端完成")
else:
    mqtt_send_queue = FakeQueue()
    logger.warning("使用假的MQTT客户端")
# -------------------- 初始化MQTT客户端 -------------------- #

# 设备启动消息
logger.info("send device startup message")
send_msg = {
    "cmd": "devicestate",
    "body": {
        "did": MQTTConfig.getattr("did"),
        "type": "startup",
        "at": get_now_time(),
        "sw_version": "230704180",  # 版本号
        "code": 200,
        "msg": "device starting",
    },
}
mqtt_send_queue.put(send_msg)


# -------------------- 初始化全局变量 -------------------- #
logger.info("init global variables start")

save_dir: Path = MainConfig.getattr("save_dir")
location_save_dir: Path = MainConfig.getattr("location_save_dir")
camera_result_save_path: Path = MainConfig.getattr("camera_result_save_path")
history_save_path: Path = MainConfig.getattr("history_save_path")
standard_save_path: Path = MainConfig.getattr("standard_save_path")
original_config_path: Path = MainConfig.getattr(
    "original_config_path"
)  # 默认 config, 用于重置
runtime_config_path: Path = MainConfig.getattr(
    "runtime_config_path"
)  # 运行时 config, 用于临时修改配置
get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
defalut_error_distance: float = MainConfig.getattr("defalut_error_distance")

# 保存原始配置
save_config_to_yaml(config_path=original_config_path)
# 从运行时 config 加载配置
init_config_from_yaml(config_path=runtime_config_path)
logger.success("init config success")

# 周期结果
camera0_cycle_results = {}
camera1_cycle_results = {}
# 是否需要发送部署信息
need_send_device_deploying_msg = False
# 是否需要发送靶标校正响应消息
need_send_target_correction_msg = False
# 是否需要发送删除靶标响应消息
need_send_delete_target_msg = False
# 是否需要发送添加靶标响应消息
need_send_set_target_msg = False
# 是否需要发送获取状态信息
need_send_get_status_msg = False
# 温度传感器返回的温度
temperature_data = {}
# 是否需要发送温控命令
need_send_temp_control_msg = True
# 是否收到温控回复命令
received_temp_control_msg = True
# 温度是否平稳
is_temp_stable = False
# 是否需要发送进入工作状态消息
need_send_in_working_state_msg = False

logger.info("init global variables end")
# -------------------- 初始化全局变量 -------------------- #

logger.success("init end")
# ------------------------------ 初始化 ------------------------------ #


def main() -> None:
    global camera0_cycle_results
    global camera1_cycle_results

    # -------------------- 控温 -------------------- #
    # target_temperature: float = TemperatureConfig.getattr("target_temperature")
    # Send.send_temperature_control_msg(target_temperature, 0)
    # Send.send_temperature_control_msg(target_temperature, 1)
    # -------------------- 控温 -------------------- #

    # ------------------------------ 调整曝光 ------------------------------ #
    # -------------------- camera0 -------------------- #
    try:
        _, image0, image0_metadata = camera0_queue.get(timeout=get_picture_timeout)
        image_path = save_dir / "image0_default.jpg"
        save_image(image0, image_path)
        logger.info(f"save `image0 default` image to {image_path}")
    except queue.Empty:
        get_picture_timeout_process()

    logger.info("camera0 ajust exposure start")
    camera0_id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr(
        "camera0_id2boxstate"
    )
    adjust_exposure_full_res(camera0_queue, camera0_id2boxstate, True, 0)
    logger.success("camera0 ajust exposure end")
    try:
        _, image0, image0_metadata = camera0_queue.get(timeout=get_picture_timeout)
        image_path = save_dir / "image0_adjust_exposure.jpg"
        save_image(image0, image_path)
        logger.info(f"save `image0 adjust exposure` image to {image_path}")
    except queue.Empty:
        get_picture_timeout_process()
    # -------------------- camera0 -------------------- #

    # 保存运行时配置
    save_config_to_yaml(config_path=runtime_config_path)

    # -------------------- camera1 -------------------- #
    try:
        _, image1, image1_metadata = camera1_queue.get(timeout=get_picture_timeout)
        image_path = save_dir / "image1_default.jpg"
        save_image(image1, image_path)
        logger.info(f"save `image1 default` image to {image_path}")
    except queue.Empty:
        get_picture_timeout_process()

    logger.info("camera1 ajust exposure start")
    camera1_id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr(
        "camera1_id2boxstate"
    )
    adjust_exposure_full_res(camera1_queue, camera1_id2boxstate, True, 1)
    logger.success("camera1 ajust exposure end")
    try:
        _, image1, image1_metadata = camera1_queue.get(timeout=get_picture_timeout)
        image_path = save_dir / "image1_adjust_exposure.jpg"
        save_image(image1, image_path)
        logger.info(f"save `image1 adjust exposure` image to {image_path}")
    except queue.Empty:
        get_picture_timeout_process()
    # -------------------- camera1 -------------------- #
    # ------------------------------ 调整曝光 ------------------------------ #

    # 保存运行时配置
    save_config_to_yaml(config_path=runtime_config_path)

    # ------------------------------ 找到目标 ------------------------------ #
    logger.info("find target start")
    # -------------------- camera0 -------------------- #
    try:
        _, image0, _ = camera0_queue.get(timeout=get_picture_timeout)
        logger.info(
            f"{image0.shape = }, ExposureTime = {image0_metadata['ExposureTime']}, AnalogueGain = {image0_metadata['AnalogueGain']}"
        )
        # -------------------- 取图 -------------------- #

        # -------------------- 畸变矫正 -------------------- #
        logger.info("rectify image0 start")
        rectified_image0 = image0
        logger.success("rectify image0 success")
        # -------------------- 畸变矫正 -------------------- #

        # -------------------- 模板匹配 -------------------- #
        logger.info("image0 find target start")

        camera0_id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr(
            "camera0_id2boxstate"
        )
        # {
        #     i: {
        #         "ratio": ratio,
        #         "score": score,
        #         "box": box
        #     }
        # }
        if camera0_id2boxstate is None:
            logger.warning(
                "camera0_id2boxstate is None, use find_target instead of find_around_target"
            )
            camera0_id2boxstate, camera0_got_target_number = find_target(
                rectified_image0, 0
            )
        else:
            logger.success("camera0_id2boxstate is not None, use find_around_target")
            camera0_id2boxstate, camera0_got_target_number = find_around_target(
                rectified_image0, 0
            )
        logger.info(f"image0 find target camera0_id2boxstate: {camera0_id2boxstate}")
        logger.info(f"image0 find target number: {camera0_got_target_number}")

        if camera0_id2boxstate is not None:
            boxes = [
                camera0_boxestate["box"]
                for camera0_boxestate in camera0_id2boxstate.values()
                if camera0_boxestate["box"] is not None
            ]
            # 绘制boxes
            image0_draw = image0.copy()
            # image0_draw = rectified_image0.copy()
            for i in range(len(boxes)):
                cv2.rectangle(
                    img=image0_draw,
                    pt1=(boxes[i][0], boxes[i][1]),
                    pt2=(boxes[i][2], boxes[i][3]),
                    color=(255, 0, 0),
                    thickness=3,
                )
            plt.figure(figsize=(10, 10))
            plt.imshow(image0_draw, cmap="gray")
            plt.savefig(save_dir / "image0_match_template.png")
            plt.close()
        logger.success("image0 find target success")
        # -------------------- 模板匹配 -------------------- #

    except queue.Empty:
        get_picture_timeout_process()
    # -------------------- camera0 -------------------- #

    # 保存运行时配置
    save_config_to_yaml(config_path=runtime_config_path)

    # -------------------- camera1 -------------------- #
    try:
        _, image1, _ = camera1_queue.get(timeout=get_picture_timeout)
        logger.info(
            f"{image1.shape = }, ExposureTime = {image1_metadata['ExposureTime']}, AnalogueGain = {image1_metadata['AnalogueGain']}"
        )
        # -------------------- 取图 -------------------- #

        # -------------------- 畸变矫正 -------------------- #
        logger.info("rectify image1 start")
        rectified_image1 = image1
        logger.success("rectify image1 success")
        # -------------------- 畸变矫正 -------------------- #

        # -------------------- 模板匹配 -------------------- #
        logger.info("image1 find target start")

        camera1_id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr(
            "camera1_id2boxstate"
        )
        # {
        #     i: {
        #         "ratio": ratio,
        #         "score": score,
        #         "box": box
        #     }
        # }
        if camera1_id2boxstate is None:
            logger.warning(
                "camera1_id2boxstate is None, use find_target instead of find_around_target"
            )
            camera1_id2boxstate, camera1_got_target_number = find_target(
                rectified_image1, 1
            )
        else:
            logger.success("camera1_id2boxstate is not None, use find_around_target")
            camera1_id2boxstate, camera1_got_target_number = find_around_target(
                rectified_image1, 1
            )
        logger.info(f"image1 find target camera1_id2boxstate: {camera1_id2boxstate}")
        logger.info(f"image1 find target number: {camera1_got_target_number}")

        if camera1_id2boxstate is not None:
            boxes = [
                camera1_boxestate["box"]
                for camera1_boxestate in camera1_id2boxstate.values()
                if camera1_boxestate["box"] is not None
            ]
            # 绘制boxes
            image1_draw = image1.copy()
            # image0_draw = rectified_image0.copy()
            for i in range(len(boxes)):
                cv2.rectangle(
                    img=image1_draw,
                    pt1=(boxes[i][0], boxes[i][1]),
                    pt2=(boxes[i][2], boxes[i][3]),
                    color=(255, 0, 0),
                    thickness=3,
                )
            plt.figure(figsize=(10, 10))
            plt.imshow(image1_draw, cmap="gray")
            plt.savefig(save_dir / "image1_match_template.png")
            plt.close()
        logger.success("image1 find target success")
        # -------------------- 模板匹配 -------------------- #

    except queue.Empty:
        get_picture_timeout_process()
    # -------------------- camera1 -------------------- #

    logger.success("find target end")
    # ------------------------------ 找到目标 ------------------------------ #

    # 保存运行时配置
    save_config_to_yaml(config_path=runtime_config_path)

    # -------------------- 初始化周期内变量 -------------------- #
    # 主循环
    i = 0
    # 一个周期内总循环次数
    total_cycle_loop_count = 0
    # 一个周期内循环计数
    cycle_loop_count = -1
    # 每个周期的间隔时间
    cycle_time_interval: int = MainConfig.getattr("cycle_time_interval")
    cycle_before_time = time.time()

    # 是否使用补光灯
    use_flash = False
    led_level = 1
    # -------------------- 初始化周期内变量 -------------------- #

    while True:
        cycle_current_time = time.time()
        # 取整为时间周期
        _cycle_before_time_period = int(cycle_before_time * 1000 // cycle_time_interval)
        _cycle_current_time_period = int(
            cycle_current_time * 1000 // cycle_time_interval
        )
        # 进入周期
        # 条件为 当前时间周期大于等于前一个时间周期 或者 周期已经开始运行
        if (
            _cycle_current_time_period > _cycle_before_time_period
            or cycle_loop_count > -1
        ):
            # 每个周期的第一次循环
            if cycle_loop_count == -1:
                logger.success("The cycle is started.")

                # 周期结果重置
                camera0_cycle_results = {}
                camera1_cycle_results = {}

                # ------------------------- 调整全图曝光 ------------------------- #
                # --------------- camera0 --------------- #
                logger.info("full image0 ajust exposure start")

                # 最大 led level
                max_led_level: int = AdjustCameraConfig.getattr("max_led_level")
                # 每次使用补光灯调整曝光的总次数
                adjust_with_flash_total_times: int = AdjustCameraConfig.getattr(
                    "adjust_with_flash_total_times"
                )
                adjust_with_falsh_total_time = 0
                while True:
                    # 如果上一次使用了补光灯，那这一次也使用补光灯
                    if use_flash:
                        Send.send_open_led_level_msg(led_level)

                    # 调整曝光
                    camera0_id2boxstate: dict[int, dict] | None = (
                        MatchTemplateConfig.getattr("camera0_id2boxstate")
                    )
                    _, need_darker, need_lighter = adjust_exposure_full_res(
                        camera0_queue,
                        camera0_id2boxstate,
                        True,
                        0,
                    )

                    # ---------- 补光灯 ---------- #
                    if need_darker or need_lighter:
                        use_flash = True
                        if need_darker:
                            # 已经是最低的补光灯
                            if led_level <= 1:
                                # 关闭补光灯
                                use_flash = False
                                logger.warning(
                                    "already is the lowest flash, close flash"
                                )
                                Send.send_close_led_msg()
                            else:
                                # 降低补光灯亮度
                                led_level -= 1
                        else:
                            # 已经是最高的补光灯
                            if led_level >= max_led_level:
                                logger.warning(
                                    "already is the highest flash, can't adjust flash"
                                )
                                continue
                            else:
                                # 增加补光灯亮度
                                led_level += 1
                    else:
                        logger.success("no need adjust flash, exit adjust exposure")
                        break

                    adjust_with_falsh_total_time += 1
                    if adjust_with_falsh_total_time >= adjust_with_flash_total_times:
                        logger.warning(
                            f"adjust exposure failed in {adjust_with_flash_total_times} times, use last result"
                        )
                        break
                    # ---------- 补光灯 ---------- #
                logger.info("full image0 ajust exposure end")
                # --------------- camera0 --------------- #

                # --------------- camera1 --------------- #
                camera1_id2boxstate: dict[int, dict] | None = (
                    MatchTemplateConfig.getattr("camera1_id2boxstate")
                )
                adjust_exposure_full_res(
                    camera1_queue,
                    camera1_id2boxstate,
                    True,
                    1,
                )
                # --------------- camera1 --------------- #
                # ------------------------- 调整全图曝光 ------------------------- #

                # ------------------------- 小区域模板匹配 + 调整 box 曝光 ------------------------- #

                try:
                    # -------------------- camera0 -------------------- #
                    drop_excessive_queue_items(camera0_queue)
                    _, image0, _ = camera0_queue.get(timeout=get_picture_timeout)

                    # --------------- 畸变矫正 --------------- #
                    rectified_image0 = image0
                    # --------------- 畸变矫正 --------------- #

                    # --------------- 小区域模板匹配 --------------- #
                    _, camera0_got_target_number = find_around_target(
                        rectified_image0, 0
                    )
                    if camera0_got_target_number == 0:
                        # ⚠️⚠️⚠️ 本次循环没有找到目标 ⚠️⚠️⚠️
                        logger.warning(
                            "camera0 find_around_target failed, can't find any target"
                        )
                    # --------------- 小区域模板匹配 --------------- #

                    # --------------- 调整 box 曝光 --------------- #
                    logger.info("camera0 boxes ajust exposure start")

                    # camera0_id2boxstate example: {
                    #     0: {'ratio': 0.8184615384615387, 'score': 0.92686927318573, 'box': [1509, 967, 1828, 1286]}},
                    #     1: {'ratio': 1.2861538461538469, 'score': 0.8924368023872375, 'box': [1926, 1875, 2427, 2376]}
                    # }
                    camera0_id2boxstate: dict[int, dict] | None = (
                        MatchTemplateConfig.getattr("camera0_id2boxstate")
                    )
                    # exposure2camera0_id2boxstate example: {
                    #     60000: {0: {'ratio': 0.8184615384615387, 'score': 0.92686927318573, 'box': [1509, 967, 1828, 1286]}},
                    #     62000: {1: {'ratio': 1.2861538461538469, 'score': 0.8924368023872375, 'box': [1926, 1875, 2427, 2376]}}
                    # }
                    # camera0_id2boxstate 为 None 时，理解为没有任何 box，调整曝光时设定为 {}
                    exposure2camera0_id2boxstate, _, _ = adjust_exposure_full_res(
                        camera0_queue,
                        camera0_id2boxstate if camera0_id2boxstate is not None else {},
                        False,
                        0,
                    )
                    logger.info(f"{exposure2camera0_id2boxstate = }")

                    # camera0 曝光时间列表
                    camera0_cycle_exposure_times = list(
                        exposure2camera0_id2boxstate.keys()
                    )
                    # index, exposure_time
                    camera0_cycle_exposure_times = [
                        (0, t) for t in camera0_cycle_exposure_times
                    ]
                    logger.info("camera0 boxes ajust exposure end")
                    # --------------- 调整 box 曝光 --------------- #
                    # -------------------- camera0 -------------------- #

                    # -------------------- camera1 -------------------- #
                    drop_excessive_queue_items(camera1_queue)
                    _, image1, _ = camera1_queue.get(timeout=get_picture_timeout)

                    # --------------- 畸变矫正 --------------- #
                    rectified_image1 = image1
                    # --------------- 畸变矫正 --------------- #

                    # --------------- 小区域模板匹配 --------------- #
                    _, camera1_got_target_number = find_around_target(
                        rectified_image1, 1
                    )
                    if camera1_got_target_number == 0:
                        # ⚠️⚠️⚠️ 本次循环没有找到目标 ⚠️⚠️⚠️
                        logger.warning(
                            "camera1 find_around_target failed, can't find any target"
                        )
                    # --------------- 小区域模板匹配 --------------- #

                    # --------------- 调整 box 曝光 --------------- #
                    logger.info("camera1 boxes ajust exposure start")

                    # camera1_id2boxstate example: {
                    #     0: {'ratio': 0.8184615384615387, 'score': 0.92686927318573, 'box': [1509, 967, 1828, 1286]}},
                    #     1: {'ratio': 1.2861538461538469, 'score': 0.8924368023872375, 'box': [1926, 1875, 2427, 2376]}
                    # }
                    camera1_id2boxstate: dict[int, dict] | None = (
                        MatchTemplateConfig.getattr("camera1_id2boxstate")
                    )
                    # exposure2camera1_id2boxstate example: {
                    #     60000: {0: {'ratio': 0.8184615384615387, 'score': 0.92686927318573, 'box': [1509, 967, 1828, 1286]}},
                    #     62000: {1: {'ratio': 1.2861538461538469, 'score': 0.8924368023872375, 'box': [1926, 1875, 2427, 2376]}}
                    # }
                    # camera1_id2boxstate 为 None 时，理解为没有任何 box，调整曝光时设定为 {}
                    exposure2camera1_id2boxstate, _, _ = adjust_exposure_full_res(
                        camera1_queue,
                        camera1_id2boxstate if camera1_id2boxstate is not None else {},
                        False,
                        1,
                    )
                    logger.info(f"{exposure2camera1_id2boxstate = }")

                    # camera1 曝光时间列表
                    camera1_cycle_exposure_times = list(
                        exposure2camera1_id2boxstate.keys()
                    )
                    # index, exposure_time
                    camera1_cycle_exposure_times = [
                        (1, t) for t in camera1_cycle_exposure_times
                    ]
                    logger.info("camera1 boxes ajust exposure end")
                    # --------------- 调整 box 曝光 --------------- #
                    # -------------------- camera1 -------------------- #

                    # -------------------- 设定循环参数 -------------------- #
                    # 所有相机循环曝光时间
                    cycle_exposure_times = (
                        camera0_cycle_exposure_times + camera1_cycle_exposure_times
                    )
                    logger.info(f"{cycle_exposure_times = }")

                    # 总的循环轮数为 1 + 曝光次数
                    cycle_exposure_times_len = len(cycle_exposure_times)
                    total_cycle_loop_count = (
                        1 + cycle_exposure_times_len if cycle_exposure_times_len else 2
                    )
                    logger.success(
                        f"During this cycle, there will be {total_cycle_loop_count} iters."
                    )
                    # 当前周期，采用从 0 开始
                    cycle_loop_count = 0
                    logger.info(f"The {cycle_loop_count} iter within the cycle.")

                    # 设置下一轮的曝光值
                    if len(cycle_exposure_times):
                        camera_index, exposure_time = cycle_exposure_times[
                            cycle_loop_count
                        ]
                        if camera_index == 0:
                            CameraConfig.setattr("camera0_exposure_time", exposure_time)
                        else:
                            CameraConfig.setattr("camera1_exposure_time", exposure_time)
                    # -------------------- 设定循环参数 -------------------- #

                    # 周期设置
                    cycle_before_time = cycle_current_time

                except queue.Empty:
                    get_picture_timeout_process()
                # ------------------------- 小区域模板匹配 + 调整 box 曝光 ------------------------- #

            # 每个周期的其余循环
            else:
                # -------------------- 获取图片 -------------------- #
                logger.info(f"The {cycle_loop_count + 1} iter within the cycle.")

                if len(cycle_exposure_times):
                    # -------------------- box location -------------------- #
                    camera_index, exposure_time = cycle_exposure_times[cycle_loop_count]

                    if camera_index == 0:
                        try:
                            # 忽略多余的图片
                            drop_excessive_queue_items(camera0_queue)

                            # 获取照片
                            image0_timestamp, image0, image0_metadata = (
                                camera0_queue.get(timeout=get_picture_timeout)
                            )
                            logger.info(
                                f"camera0 get image: {image0_timestamp}, ExposureTime = {image0_metadata['ExposureTime']}, AnalogueGain = {image0_metadata['AnalogueGain']}, shape = {image0.shape}"
                            )

                            # -------------------- 畸变矫正 -------------------- #
                            rectified_image0 = image0
                            # -------------------- 畸变矫正 -------------------- #

                            # -------------------- 检测 -------------------- #
                            camera0_id2boxstate = exposure2camera0_id2boxstate[
                                exposure_time
                            ]
                            logger.info(
                                f"cycle_loop_count: {cycle_loop_count}, camera0, {exposure_time = }, {camera0_id2boxstate = }"
                            )
                            for (
                                box_id,
                                camera0_boxestate,
                            ) in camera0_id2boxstate.items():
                                logger.info("camera0 box location start")
                                camera0_rings_location_result = rings_location(
                                    rectified_image0,
                                    box_id,
                                    camera0_boxestate,
                                    image0_timestamp,
                                    image0_metadata,
                                )
                                camera0_cycle_results[box_id] = (
                                    camera0_rings_location_result
                                )
                                logger.info(
                                    f"camera0 ring location result: {camera0_rings_location_result}"
                                )
                            # -------------------- 检测 -------------------- #

                            # 没有发生错误, 周期内循环计数加1
                            cycle_loop_count += 1

                        except queue.Empty:
                            get_picture_timeout_process()
                    else:
                        try:
                            # 忽略多余的图片
                            drop_excessive_queue_items(camera1_queue)

                            # 获取照片
                            image1_timestamp, image1, image1_metadata = (
                                camera1_queue.get(timeout=get_picture_timeout)
                            )
                            logger.info(
                                f"camera1 get image: {image1_timestamp}, ExposureTime = {image1_metadata['ExposureTime']}, AnalogueGain = {image1_metadata['AnalogueGain']}, shape = {image1.shape}"
                            )

                            # -------------------- 畸变矫正 -------------------- #
                            rectified_image1 = image1
                            # -------------------- 畸变矫正 -------------------- #

                            # -------------------- 检测 -------------------- #
                            camera1_id2boxstate = exposure2camera1_id2boxstate[
                                exposure_time
                            ]
                            logger.info(
                                f"cycle_loop_count: {cycle_loop_count}, camera1, {exposure_time = }, {camera1_id2boxstate = }"
                            )
                            for (
                                box_id,
                                camera1_boxestate,
                            ) in camera1_id2boxstate.items():
                                logger.info("camera1 box location start")
                                camera1_rings_location_result = rings_location(
                                    rectified_image1,
                                    box_id,
                                    camera1_boxestate,
                                    image1_timestamp,
                                    image1_metadata,
                                )
                                camera1_cycle_results[box_id] = (
                                    camera1_rings_location_result
                                )
                                logger.info(
                                    f"camera1 ring location result: {camera1_rings_location_result}"
                                )
                            # -------------------- 检测 -------------------- #

                            # 没有发生错误, 周期内循环计数加1
                            cycle_loop_count += 1

                        except queue.Empty:
                            get_picture_timeout_process()
                    # -------------------- box location -------------------- #
                else:
                    # 没有需要曝光的图片, cycle_loop_count 也要加1, 因为默认的总循环次数为2
                    cycle_loop_count += 1

                # 正常判断是否结束周期
                if cycle_loop_count >= total_cycle_loop_count - 1:
                    # ------------------------- 整理检测结果 ------------------------- #
                    logger.info("last cycle, try to compare and save results")
                    logger.success(f"{camera0_cycle_results = }")
                    logger.success(f"{camera1_cycle_results = }")
                    # 保存到文件
                    save_to_jsonl(
                        {
                            "camera0": camera0_cycle_results,
                            "camera1": camera1_cycle_results,
                            "temperature": temperature_data,
                        },
                        history_save_path,
                    )

                    # 防止值不存在
                    send_msg_data = {}

                    target_number: int = MatchTemplateConfig.getattr("target_number")

                    main_camera_index: int = CameraConfig.getattr("main_camera_index")
                    camera_left_index: int = CameraConfig.getattr("camera_left_index")
                    camera0_standard_results: dict | None = RingsLocationConfig.getattr(
                        "camera0_standard_results"
                    )
                    camera1_standard_results: dict | None = RingsLocationConfig.getattr(
                        "camera1_standard_results"
                    )
                    z_move_threshold: float = RingsLocationConfig.getattr(
                        "z_move_threshold"
                    )
                    # 初始化标准结果
                    if (
                        camera0_standard_results is None
                        or len(camera0_standard_results) != target_number
                        or camera1_standard_results is None
                        or len(camera1_standard_results) != target_number
                    ):
                        # -------------------- camera0 init result -------------------- #
                        logger.info("try to init camera0_standard_results")
                        camera0_reference_target_id2offset: (
                            dict[int, tuple[float, float]] | None
                        ) = RingsLocationConfig.getattr(
                            "camera0_reference_target_id2offset"
                        )

                        # 初始化标准靶标
                        camera0_standard_results = init_standard_results(
                            camera0_cycle_results,
                            camera0_standard_results,
                            camera0_reference_target_id2offset,
                        )
                        # -------------------- camera0 init result -------------------- #

                        # -------------------- camera1 init result -------------------- #
                        logger.info("try to init camera1_standard_results")
                        camera1_reference_target_id2offset: (
                            dict[int, tuple[float, float]] | None
                        ) = RingsLocationConfig.getattr(
                            "camera1_reference_target_id2offset"
                        )

                        # 初始化标准靶标
                        camera1_standard_results = init_standard_results(
                            camera1_cycle_results,
                            camera1_standard_results,
                            camera1_reference_target_id2offset,
                        )
                        # -------------------- camera1 init result -------------------- #

                        # -------------------- send result -------------------- #
                        if (
                            camera0_standard_results is not None
                            and camera1_standard_results is not None
                        ):
                            RingsLocationConfig.setattr(
                                "camera0_standard_results", camera0_standard_results
                            )
                            RingsLocationConfig.setattr(
                                "camera1_standard_results", camera1_standard_results
                            )
                            # 发送初始坐标数据结果
                            # ✅️✅️✅️ 正常数据消息 ✅️✅️✅️
                            logger.success("send init data message.")
                            send_msg_data = deepcopy(temperature_data)
                            camera0_send_msg_data = {
                                f"L1_SJ_{k+1}": {"X": 0, "Y": 0}
                                for k in camera0_standard_results.keys()
                            }
                            camera1_send_msg_data = {
                                f"L1_SJ_{k+1}": {"X": 0, "Y": 0}
                                for k in camera0_standard_results.keys()
                            }
                            src_data = {
                                "left_cam": camera0_send_msg_data
                                if camera_left_index == 0
                                else camera1_send_msg_data,
                                "right_cam": camera1_send_msg_data
                                if camera_left_index == 0
                                else camera0_send_msg_data,
                            }

                            _send_msg_data = (
                                deepcopy(camera0_send_msg_data)
                                if main_camera_index == 0
                                else deepcopy(camera1_send_msg_data)
                            )
                            # add z axis
                            _send_msg_data = {
                                k: {**v, "Z": 0} for k, v in _send_msg_data.items()
                            }
                            _send_msg_data["src_data"] = src_data

                            send_msg_data.update(_send_msg_data)
                            logger.info(f"send_msg_data: {send_msg_data}")
                            send_msg = {
                                "cmd": "visiondp",
                                "did": MQTTConfig.getattr("did"),
                                "at": get_now_time(),
                                "data": send_msg_data,
                            }
                            mqtt_send_queue.put(send_msg)
                        # -------------------- send result -------------------- #

                    # 比较标准靶标和新的靶标
                    else:
                        # -------------------- camera0 compare results -------------------- #
                        logger.info("try to compare camera0 result")
                        camera0_reference_target_id2offset: (
                            dict[int, tuple[float, float]] | None
                        ) = RingsLocationConfig.getattr(
                            "camera0_reference_target_id2offset"
                        )

                        # 计算距离
                        (
                            camera0_distance_result,
                            camera0_over_distance_ids,
                            camera0_reference_target_id2offset,
                        ) = calc_move_distance(
                            camera0_standard_results,
                            camera0_cycle_results,
                            camera0_reference_target_id2offset,
                        )
                        # 更新参考靶标偏移
                        RingsLocationConfig.setattr(
                            "camera0_reference_target_id2offset",
                            camera0_reference_target_id2offset,
                        )
                        logger.info(
                            f"camera0_distance_result: {camera0_distance_result}"
                        )
                        logger.info(
                            f"camera0_over_distance_ids: {camera0_over_distance_ids}"
                        )
                        # -------------------- camera0 compare results -------------------- #

                        # -------------------- camera1 compare results -------------------- #
                        logger.info("try to compare camera1 result")
                        camera1_reference_target_id2offset: (
                            dict[int, tuple[float, float]] | None
                        ) = RingsLocationConfig.getattr(
                            "camera1_reference_target_id2offset"
                        )

                        # 计算距离
                        (
                            camera1_distance_result,
                            camera1_over_distance_ids,
                            camera1_reference_target_id2offset,
                        ) = calc_move_distance(
                            camera1_standard_results,
                            camera1_cycle_results,
                            camera1_reference_target_id2offset,
                        )
                        # 更新参考靶标偏移
                        RingsLocationConfig.setattr(
                            "camera1_reference_target_id2offset",
                            camera1_reference_target_id2offset,
                        )
                        logger.info(
                            f"camera1_distance_result: {camera1_distance_result}"
                        )
                        logger.info(
                            f"camera1_over_distance_ids: {camera1_over_distance_ids}"
                        )
                        # -------------------- camera1 compare results -------------------- #

                        # -------------------- send msg -------------------- #

                        send_msg_data = deepcopy(temperature_data)
                        camera0_send_msg_data = {
                            f"L1_SJ_{k+1}": {"X": v[0], "Y": v[1]}
                            for k, v in camera0_distance_result.items()
                        }
                        camera1_send_msg_data = {
                            f"L1_SJ_{k+1}": {"X": v[0], "Y": v[1]}
                            for k, v in camera1_distance_result.items()
                        }
                        src_data = {
                            "left_cam": camera0_send_msg_data
                            if camera_left_index == 0
                            else camera1_send_msg_data,
                            "right_cam": camera1_send_msg_data
                            if camera_left_index == 0
                            else camera0_send_msg_data,
                        }

                        _send_msg_data = (
                            deepcopy(camera0_send_msg_data)
                            if main_camera_index == 0
                            else deepcopy(camera1_send_msg_data)
                        )
                        # TODO: 求 Z 轴的变化, 临时使用主相机的检测距离计算
                        if main_camera_index == 0:
                            temp_standard_results = camera0_standard_results
                            temp_cycle_results = camera0_cycle_results
                        else:
                            temp_standard_results = camera1_standard_results
                            temp_cycle_results = camera1_cycle_results
                        ndigits: int = RingsLocationConfig.getattr("ndigits")
                        z_move = {
                            f"L1_SJ_{k+1}": round(
                                temp_cycle_results[k]["distance"]
                                - temp_standard_results[k]["distance"],
                                ndigits,
                            )
                            if (
                                temp_cycle_results[k]["distance"] is not None
                                and temp_standard_results[k]["distance"] is not None
                            )
                            else defalut_error_distance
                            for k in temp_standard_results.keys()
                        }
                        logger.info(f"z_move: {z_move}")
                        # 计算 z 轴是否超出阈值
                        z_over_distance_ids = []
                        for k, v in z_move.items():
                            if abs(v) > z_move_threshold:
                                z_over_distance_ids.append(int(k.split("_")[-1]) - 1)
                                logger.warning(
                                    f"box {k} z move distance is over threshold {z_move_threshold}."
                                )

                        _send_msg_data = {
                            k: {**v, "Z": z_move[k]} for k, v in _send_msg_data.items()
                        }
                        _send_msg_data["src_data"] = src_data
                        send_msg_data.update(_send_msg_data)
                        logger.info(f"send_msg_data: {send_msg_data}")

                        # 超出阈值id
                        over_distance_ids = (
                            deepcopy(camera0_over_distance_ids)
                            if main_camera_index == 0
                            else deepcopy(camera1_over_distance_ids)
                        )
                        # 添加 z 轴超出阈值 id
                        for z_over_distance_id in z_over_distance_ids:
                            over_distance_ids.add(z_over_distance_id)
                        if len(over_distance_ids) > 0:
                            # ⚠️⚠️⚠️ 有box移动距离超过阈值 ⚠️⚠️⚠️
                            logger.warning(
                                f"box {over_distance_ids} move distance is over threshold."
                            )

                            # 保存位移的图片
                            image_path0 = save_dir / "target_displacement0.jpg"
                            image_path1 = save_dir / "target_displacement1.jpg"
                            save_image(image0, image_path0)
                            save_image(image1, image_path1)
                            logger.info(
                                f"save `target displacement` image to {image_path0}, {image_path1}"
                            )
                            # 位移告警消息
                            send_msg = {
                                "cmd": "alarm",
                                "body": {
                                    "did": MQTTConfig.getattr("did"),
                                    "type": "displacement",
                                    "at": get_now_time(),
                                    "number": [
                                        i + 1 for i in over_distance_ids
                                    ],  # 表示异常的靶标编号
                                    "data": send_msg_data,
                                    "path": [
                                        str(image_path0),
                                        str(image_path1),
                                    ],  # 图片本地路径
                                    "img": [
                                        "target_displacement0.jpg",
                                        "target_displacement1.jpg",
                                    ],  # 文件名称
                                },
                            }
                            mqtt_send_queue.put(send_msg)
                        else:
                            # ✅️✅️✅️ 所有 box 移动距离都小于阈值 ✅️✅️✅️
                            logger.success("All box move distance is under threshold.")
                            # ✅️✅️✅️ 正常数据消息 ✅️✅️✅️
                            send_msg = {
                                "cmd": "visiondp",
                                "did": MQTTConfig.getattr("did"),
                                "at": get_now_time(),
                                "data": send_msg_data,
                            }
                            mqtt_send_queue.put(send_msg)
                        # -------------------- send msg -------------------- #

                    # ------------------------- 整理检测结果 ------------------------- #

                    # ------------------------- 检查是否丢失目标 ------------------------- #
                    target_number = MatchTemplateConfig.getattr("target_number")

                    # -------------------- camera0 lost box -------------------- #
                    camera0_got_target_number: int = MatchTemplateConfig.getattr(
                        "camera0_got_target_number"
                    )
                    if target_number > camera0_got_target_number or target_number == 0:
                        logger.warning(
                            f"camera0 target number {target_number} is not enough, got {camera0_got_target_number} targets, start to find lost target."
                        )

                        # 忽略多余的图片
                        drop_excessive_queue_items(camera0_queue)

                        try:
                            # 获取照片
                            _, image0, _ = camera0_queue.get(
                                timeout=get_picture_timeout
                            )

                            # -------------------- 畸变矫正 -------------------- #
                            rectified_image0 = image0
                            # -------------------- 畸变矫正 -------------------- #

                            # -------------------- 模板匹配 -------------------- #
                            find_lost_target(rectified_image0, 0)
                            target_number = MatchTemplateConfig.getattr("target_number")
                            camera0_got_target_number = MatchTemplateConfig.getattr(
                                "camera0_got_target_number"
                            )
                            # -------------------- 模板匹配 -------------------- #

                            if (
                                target_number > camera0_got_target_number
                                or target_number == 0
                            ):
                                # ❌️❌️❌️ 重新查找完成之后仍然不够 ❌️❌️❌️
                                # 获取丢失的box idx
                                camera0_id2boxstate: dict[int, dict] | None = (
                                    MatchTemplateConfig.getattr("camera0_id2boxstate")
                                )
                                if camera0_id2boxstate is not None:
                                    loss_ids = [
                                        i
                                        for i, camera0_boxestate in camera0_id2boxstate.items()
                                        if camera0_boxestate["box"] is None
                                    ]
                                else:
                                    # 假如开始没有任何 box, 则认为丢失的 box idx 为 []
                                    loss_ids = []

                                logger.critical(
                                    f"camera0 target number {target_number} is not enough, got {camera0_got_target_number} targets, loss box ids: {loss_ids}."
                                )

                                # 保存丢失的图片
                                image_path = save_dir / "target_loss0.jpg"
                                save_image(image0, image_path)
                                logger.info(f"save `target loss` image to {image_path}")
                                # 目标丢失告警消息
                                send_msg = {
                                    "cmd": "alarm",
                                    "body": {
                                        "did": MQTTConfig.getattr("did"),
                                        "type": "target_loss",
                                        "at": get_now_time(),
                                        "number": [
                                            i + 1 for i in loss_ids
                                        ],  # 异常的靶标编号
                                        "data": send_msg_data,
                                        "path": [str(image_path)],  # 图片本地路径
                                        "img": ["target_loss0.jpg"],  # 文件名称
                                    },
                                }
                                mqtt_send_queue.put(send_msg)
                            else:
                                # ✅️✅️✅️ 丢失目标重新找回 ✅️✅️✅️
                                logger.success(
                                    f"camera0 lost target has been found, the target number {target_number} is enough, got {camera0_got_target_number} targets."
                                )

                        except queue.Empty:
                            get_picture_timeout_process()

                    # 目标数量正常
                    else:
                        logger.success(
                            f"camera0 target number {target_number} is enough, got {camera0_got_target_number} targets."
                        )
                    # -------------------- camera0 lost box -------------------- #

                    # -------------------- camera1 lost box -------------------- #
                    camera1_got_target_number = MatchTemplateConfig.getattr(
                        "camera1_got_target_number"
                    )
                    if target_number > camera1_got_target_number or target_number == 0:
                        logger.warning(
                            f"camera1 target number {target_number} is not enough, got {camera1_got_target_number} targets, start to find lost target."
                        )

                        # 忽略多余的图片
                        drop_excessive_queue_items(camera1_queue)

                        try:
                            # 获取照片
                            _, image1, _ = camera1_queue.get(
                                timeout=get_picture_timeout
                            )

                            # -------------------- 畸变矫正 -------------------- #
                            rectified_image1 = image1
                            # -------------------- 畸变矫正 -------------------- #

                            # -------------------- 模板匹配 -------------------- #
                            find_lost_target(rectified_image1, 1)
                            target_number = MatchTemplateConfig.getattr("target_number")
                            camera1_got_target_number = MatchTemplateConfig.getattr(
                                "camera1_got_target_number"
                            )
                            # -------------------- 模板匹配 -------------------- #

                            if (
                                target_number > camera1_got_target_number
                                or target_number == 0
                            ):
                                # ❌️❌️❌️ 重新查找完成之后仍然不够 ❌️❌️❌️
                                # 获取丢失的box idx
                                camera1_id2boxstate: dict[int, dict] | None = (
                                    MatchTemplateConfig.getattr("camera1_id2boxstate")
                                )
                                if camera1_id2boxstate is not None:
                                    loss_ids = [
                                        i
                                        for i, camera1_boxestate in camera1_id2boxstate.items()
                                        if camera1_boxestate["box"] is None
                                    ]
                                else:
                                    # 假如开始没有任何 box, 则认为丢失的 box idx 为 []
                                    loss_ids = []

                                logger.critical(
                                    f"camera1 target number {target_number} is not enough, got {camera1_got_target_number} targets, loss box ids: {loss_ids}."
                                )

                                # 保存丢失的图片
                                image_path = save_dir / "target_loss1.jpg"
                                save_image(image1, image_path)
                                logger.info(f"save `target loss` image to {image_path}")
                                # 目标丢失告警消息
                                send_msg = {
                                    "cmd": "alarm",
                                    "body": {
                                        "did": MQTTConfig.getattr("did"),
                                        "type": "target_loss",
                                        "at": get_now_time(),
                                        "number": [
                                            i + 1 for i in loss_ids
                                        ],  # 异常的靶标编号
                                        "data": send_msg_data,
                                        "path": [str(image_path)],  # 图片本地路径
                                        "img": ["target_loss1.jpg"],  # 文件名称
                                    },
                                }
                                mqtt_send_queue.put(send_msg)
                            else:
                                # ✅️✅️✅️ 丢失目标重新找回 ✅️✅️✅️
                                logger.success(
                                    f"camera1 lost target has been found, the target number {target_number} is enough, got {camera1_got_target_number} targets."
                                )

                        except queue.Empty:
                            get_picture_timeout_process()

                    # 目标数量正常
                    else:
                        logger.success(
                            f"camera1 target number {target_number} is enough, got {camera1_got_target_number} targets."
                        )
                    # -------------------- camera1 lost box -------------------- #
                    # ------------------------- 检查是否丢失目标 ------------------------- #

                    # ------------------------- 结束周期 ------------------------- #
                    # 重置周期内循环计数
                    cycle_loop_count = -1

                    if use_flash:
                        # 关闭补光灯
                        Send.send_close_led_msg()

                    logger.success("The cycle is over.")
                    # ------------------------- 结束周期 ------------------------- #
                else:
                    # 不是结束周期，设置下一轮的曝光值
                    camera_index, exposure_time = cycle_exposure_times[cycle_loop_count]
                    if camera_index == 0:
                        CameraConfig.setattr("camera0_exposure_time", exposure_time)
                    else:
                        CameraConfig.setattr("camera1_exposure_time", exposure_time)

        # 检测周期外
        if cycle_loop_count == -1:
            # ------------------------- 获取消息 ------------------------- #
            if not main_queue.empty():
                Receive.main_queue_receive_msg()
            # ------------------------- 获取消息 ------------------------- #

            # ------------------------- 发送消息 ------------------------- #
            Send.main_send_msg()
            # ------------------------- 发送消息 ------------------------- #

            # 保存运行时配置
            save_config_to_yaml(config_path=runtime_config_path)
            # 重新获取周期时间
            cycle_time_interval: int = MainConfig.getattr("cycle_time_interval")

        # 主循环休眠
        main_sleep_interval: int = MainConfig.getattr("main_sleep_interval")
        time.sleep(main_sleep_interval / 1000)

        # 测试调整相机
        if i > 5000:
            os._exit(0)
        logger.debug(f"{i = }")
        i += 1


class Receive:
    @staticmethod
    def main_queue_receive_msg():
        while not main_queue.empty():
            received_msg: dict = main_queue.get()
            Receive.switch(received_msg)

    @staticmethod
    def switch(received_msg: dict):
        cmd = received_msg.get("cmd")
        logger.info(f"received msg: {received_msg}")

        # 设备部署消息
        if cmd == "devicedeploying":
            Receive.receive_device_deploying_msg(received_msg)

        # # 靶标校正消息
        # elif cmd == "targetcorrection":
        #     Receive.receive_target_correction_msg(received_msg)

        # elif cmd == "deletetarget":
        #     Receive.receive_delete_target_msg(received_msg)

        # elif cmd == "settarget":
        #     Receive.receive_set_target_msg(received_msg)

        # # 参考靶标设定消息
        # elif cmd == "setreferencetarget":
        #     Receive.receive_set_reference_target_msg(received_msg)

        # 设备状态查询消息
        elif cmd == "getstatus":
            Receive.receive_getstatus_msg(received_msg)

        # 现场图像查询消息
        elif cmd == "getimage":
            Receive.receive_get_image_msg(received_msg)

        # 温控板回复控温指令, 回复可能延期
        elif cmd == "askadjusttempdata":
            Receive.receive_temp_control_msg(received_msg)

        # 日常温度数据
        elif cmd == "sendtempdata":
            Receive.receive_temp_data_msg(received_msg)

        # 温度调节过程数据
        elif cmd == "sendadjusttempdata":
            Receive.receive_adjust_temp_data_msg(received_msg)

        # 温控停止消息
        elif cmd == "stopadjusttemp":
            Receive.receive_stop_adjust_temp_data(received_msg)

        # 重启终端设备消息
        elif cmd == "reboot":
            Receive.receive_reboot_msg(received_msg)

        # 更新配置文件消息
        elif cmd == "updateconfigfile":
            Receive.receive_update_config_file_msg(received_msg)

        # 查询配置文件消息
        elif cmd == "getconfigfile":
            Receive.receive_get_config_file_msg(received_msg)

        elif cmd == "askadjustLEDlevel":
            Receive.receive_ask_adjust_led_level_with_time_msg(received_msg)

        elif cmd == "askopenLED":
            Receive.receive_ask_open_led_level_msg(received_msg)

        elif cmd == "askcloseLED":
            Receive.receive_ask_close_led_msg(received_msg)

        else:
            logger.warning(f"unknown cmd: {cmd}")
            logger.warning(f"unknown msg: {received_msg}")

    @staticmethod
    def receive_device_deploying_msg(received_msg: dict | None = None):
        """设备部署消息"""
        # {
        #     "cmd":"devicedeploying",
        #     "msgid":"bb6f3eeb2",
        # }
        global need_send_device_deploying_msg
        global camera0_cycle_results, camera1_cycle_results

        logger.info("device deploying, reset config and init target")
        # 设备部署，重置配置和初始靶标
        load_config_from_yaml(config_path=original_config_path)
        # 重设检测的结果(用于发送新的消息)
        camera0_cycle_results = {}
        camera1_cycle_results = {}
        # 需要发送部署相应消息
        need_send_device_deploying_msg = True
        logger.success(
            "device deploying success, reset config and init target, reset camera0_cycle_results"
        )

    @staticmethod
    def receive_target_correction_msg(received_msg: dict | None = None):
        """靶标校正消息"""
        global need_send_target_correction_msg
        global camera0_cycle_results

        # {
        #     "cmd":"targetcorrection",
        #     "msgid":"bb6f3eeb2",
        #     "body":{
        #         "add_boxes":[
        #             [x1, y1, x2, y2],
        #             [x1, y1, x2, y2],
        #         ],
        #         "remove_box_ids": ["L1_SJ_3", "L1_SJ_4"]
        #     }
        # }
        logger.info("target correction")
        # camera0_id2boxstate: {
        #     i: {
        #         "ratio": ratio,
        #         "score": score,
        #         "box": box
        #     }
        # }
        camera0_id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr(
            "camera0_id2boxstate"
        )
        remove_box_ids: list[str] = received_msg["body"].get("remove_box_ids", [])
        logger.info(f"remove_box_ids: {remove_box_ids}")
        # -1 因为 id 从 0 开始
        _remove_box_ids: list[int] = [
            int(remove_box_id.split("_")[-1]) - 1 for remove_box_id in remove_box_ids
        ]
        logger.info(f"int(remove_box_ids): {_remove_box_ids}")

        # 去除多余的 box
        for remove_box_id in _remove_box_ids:
            if remove_box_id in camera0_id2boxstate.keys():
                logger.info(
                    f"remove box {remove_box_id}, boxstate: {camera0_id2boxstate[remove_box_id]}"
                )
                camera0_id2boxstate.pop(remove_box_id)
            else:
                logger.warning(f"box {remove_box_id} not found in camera0_id2boxstate.")

        new_boxes: list[list[int]] = received_msg["body"].get("add_boxes", [])
        logger.info(f"new_boxes: {new_boxes}")
        # 将新的 box 转换为列表
        new_boxstates = [
            {"ratio": None, "score": None, "box": new_box} for new_box in new_boxes
        ]
        # 旧的 box 也转换为列表，并合并新 box
        new_boxstates.extend(camera0_id2boxstate.values())

        # 按照 box 排序
        new_boxes: np.ndarray = np.array(
            [boxstate["box"] for boxstate in new_boxstates]
        )
        new_ratios: np.ndarray = np.array(
            [boxstate["ratio"] for boxstate in new_boxstates]
        )
        new_scores: np.ndarray = np.array(
            [boxstate["score"] for boxstate in new_boxstates]
        )
        sorted_index = sort_boxes_center(new_boxes, sort_by="y")
        sorted_ratios = new_ratios[sorted_index]
        sorted_scores = new_scores[sorted_index]
        sorted_boxes = new_boxes[sorted_index]

        # 合并后的 box 生成新的 camera0_id2boxstate
        new_camera0_id2boxstate = {}
        for i, (ratio, score, box) in enumerate(
            zip(sorted_ratios, sorted_scores, sorted_boxes)
        ):
            new_camera0_id2boxstate[i] = {
                "ratio": float(ratio) if ratio is not None else None,
                "score": float(score) if score is not None else None,
                "box": box.tolist(),
            }
        target_number = len(new_camera0_id2boxstate)
        logger.info(f"new_id2boxstate: {new_camera0_id2boxstate}")
        logger.info(f"target_number: {target_number}")

        # 设置新目标数量和靶标信息
        MatchTemplateConfig.setattr("target_number", target_number)
        MatchTemplateConfig.setattr("camera0_id2boxstate", new_camera0_id2boxstate)
        # 删除参考靶标
        RingsLocationConfig.setattr("camera0_reference_target_id2offset", None)
        # 因为重设了靶标，所以需要重新初始化标准靶标
        RingsLocationConfig.setattr("camera0_standard_results", None)

        # 重设检测的结果(用于发送新的消息)
        camera0_cycle_results = {}
        # 需要发送靶标校正响应消息
        need_send_target_correction_msg = True

        logger.success("target correction success")

    @staticmethod
    def receive_delete_target_msg(received_msg: dict | None = None):
        """删除靶标消息"""
        global need_send_delete_target_msg
        global camera0_cycle_results
        # {
        #     "cmd": "deletetarget",
        #     "msgid": "bb6f3eeb2",
        #     "body": {
        #         "remove_box_ids": ["L1_SJ_3", "L1_SJ_4"]
        #     }
        # }
        logger.info("delete target")
        # camera0_id2boxstate: {
        #     i: {
        #         "ratio": ratio,
        #         "score": score,
        #         "box": box
        #     }
        # }
        camera0_id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr(
            "camera0_id2boxstate"
        )
        camera0_standard_results: dict | None = RingsLocationConfig.getattr(
            "camera0_standard_results"
        )

        remove_box_ids: list[str] = received_msg["body"].get("remove_box_ids", [])
        logger.info(f"remove_box_ids: {remove_box_ids}")
        # -1 因为 id 从 0 开始
        _remove_box_ids: list[int] = [
            int(remove_box_id.split("_")[-1]) - 1 for remove_box_id in remove_box_ids
        ]
        logger.info(f"int(remove_box_ids): {_remove_box_ids}")

        # 去除多余的 box
        for remove_box_id in _remove_box_ids:
            if (
                camera0_id2boxstate is not None
                and remove_box_id in camera0_id2boxstate.keys()
            ):
                logger.info(
                    f"remove box {remove_box_id}, boxstate: {camera0_id2boxstate[remove_box_id]}"
                )
                camera0_id2boxstate.pop(remove_box_id)
            else:
                logger.warning(
                    f"box {remove_box_id} not found in camera0_id2boxstate or camera0_id2boxstate is None."
                )

            if (
                camera0_standard_results is not None
                and remove_box_id in camera0_standard_results.keys()
            ):
                logger.info(
                    f"remove box {remove_box_id}, camera0_standard_results: {camera0_standard_results[remove_box_id]}"
                )
                # 因为重设了靶标，所以需要删除部分初始化标准靶标
                camera0_standard_results.pop(remove_box_id)
            else:
                logger.warning(
                    f"box {remove_box_id} not found in camera0_standard_results or camera0_standard_results is None."
                )

        target_number = len(camera0_id2boxstate)
        logger.info(f"target_number: {target_number}")
        logger.info(f"new_camera0_id2boxstate: {camera0_id2boxstate}")
        logger.info(f"new_camera0_standard_results: {camera0_standard_results}")

        # 设置新目标数量和靶标信息
        MatchTemplateConfig.setattr("target_number", target_number)
        MatchTemplateConfig.setattr("camera0_id2boxstate", camera0_id2boxstate)
        RingsLocationConfig.setattr(
            "camera0_standard_results", camera0_standard_results
        )

        # 可能删除参考靶标
        camera0_reference_target_id2offset: dict[int, tuple[float, float]] | None = (
            RingsLocationConfig.getattr("camera0_reference_target_id2offset")
        )
        if camera0_reference_target_id2offset is not None:
            reference_target_id: int = int(
                list(camera0_reference_target_id2offset.keys())[0]
            )
            if reference_target_id in _remove_box_ids:
                RingsLocationConfig.setattr("camera0_reference_target_id2offset", None)
                logger.warning(
                    f"reference target {reference_target_id} is removed, reset camera0_reference_target_id2offset."
                )

        # 重设检测的结果(用于发送新的消息)
        camera0_cycle_results = {}
        # 需要发送删除靶标响应消息
        need_send_delete_target_msg = True

        logger.success("delete target success")

    @staticmethod
    def receive_set_target_msg(received_msg: dict | None = None):
        """添加靶标消息"""
        global need_send_set_target_msg
        global camera0_cycle_results
        # {
        #     "cmd": "settarget",
        #     "msgid": "bb6f3eeb2",
        #     "body": {
        #         "add_boxes":{
        #             "L1_SJ_3": [x1, y1, x2, y2],
        #             "L1_SJ_4": [x1, y1, x2, y2]
        #         }
        #     }
        # }
        logger.info("set target")
        # camera0_id2boxstate: {
        #     i: {
        #         "ratio": ratio,
        #         "score": score,
        #         "box": box
        #     }
        # }
        camera0_id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr(
            "camera0_id2boxstate"
        )

        new_boxes: dict[str, list[int]] = received_msg["body"].get("add_boxes", {})
        logger.info(f"new_boxes: {new_boxes}")

        for new_key, new_box in new_boxes.items():
            _new_key = int(new_key.split("_")[-1]) - 1  # -1 因为 id 从 0 开始
            if _new_key in camera0_id2boxstate.keys():
                logger.warning(
                    f"box {_new_key} already exist in camera0_id2boxstate, ignore."
                )
            else:
                new_boxstate = {"ratio": None, "score": None, "box": new_box}
                camera0_id2boxstate[_new_key] = new_boxstate
                logger.info(f"add box {_new_key}, new_box: {new_box}")

        target_number = len(camera0_id2boxstate)
        logger.info(f"new_id2boxstate: {camera0_id2boxstate}")
        logger.info(f"target_number: {target_number}")

        # 设置新目标数量和靶标信息
        MatchTemplateConfig.setattr("target_number", target_number)
        MatchTemplateConfig.setattr("camera0_id2boxstate", camera0_id2boxstate)

        # 由于添加了新的box, 因此参考靶标也会更新为最新的, 因此除了参考靶标之外的其他靶标的偏移量需要重新计算, 计算方式为原本的偏移量加上参考靶标的偏移量
        camera0_standard_results: dict | None = RingsLocationConfig.getattr(
            "camera0_standard_results"
        )
        camera0_reference_target_id2offset: dict[int, tuple[float, float]] | None = (
            RingsLocationConfig.getattr("camera0_reference_target_id2offset")
        )
        if (
            camera0_standard_results is not None
            and camera0_reference_target_id2offset is not None
        ):
            ref_id: int = int(list(camera0_reference_target_id2offset.keys())[0])
            ref_distance_x, ref_distance_y = camera0_reference_target_id2offset[ref_id]
            if (
                abs(ref_distance_x) >= defalut_error_distance
                or abs(ref_distance_y) >= defalut_error_distance
            ):
                logger.warning(
                    f"reference target {ref_id} offset is too large, can not add to other target's center offset."
                )
            else:
                for key, value in camera0_standard_results.items():
                    if key != ref_id:
                        camera0_standard_results[key]["offset"] = [
                            value["offset"][0] + ref_distance_x,
                            value["offset"][1] + ref_distance_y,
                        ]
                logger.info(
                    "use reference target to add center offset to other target's center offset."
                )
        else:
            logger.warning(
                "camera0_standard_results or camera0_reference_target_id2offset is None, can not add center offset."
            )
        RingsLocationConfig.setattr(
            "camera0_standard_results", camera0_standard_results
        )

        # 重设检测的结果(用于发送新的消息)
        camera0_cycle_results = {}
        # 需要发送添加靶标响应消息
        need_send_set_target_msg = True

        logger.success("set target success")

    @staticmethod
    def receive_set_reference_target_msg(received_msg: dict | None = None):
        """参考靶标设定消息"""
        logger.info("set reference target")
        # {
        #     "cmd":"setreferencetarget",
        #     "msgid":"bb6f3eeb2",
        #     "apikey":"e343f59e9a1b426aa435",
        #     "body":{
        #         "reference_target":"L1_SJ_1"
        #     }
        # }
        reference_target: str = received_msg["body"]["reference_target"]
        logger.success(f" try set reference reference_target: {reference_target}")
        reference_target_id = (
            int(reference_target.split("_")[-1]) - 1
        )  # -1 因为 id 从 0 开始

        camera0_id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr(
            "camera0_id2boxstate"
        )
        if reference_target_id in camera0_id2boxstate.keys():
            RingsLocationConfig.setattr(
                "camera0_reference_target_id2offset", {reference_target_id: [0, 0]}
            )
            # 参考靶标设定响应消息
            send_msg = {
                "cmd": "setreferencetarget",
                "body": {
                    "code": 200,
                    "did": MQTTConfig.getattr("did"),
                    "at": get_now_time(),
                    "reference_target": reference_target,
                    "msg": "set succeed",
                },
                "msgid": "bb6f3eeb2",
            }
            mqtt_send_queue.put(send_msg)
            logger.success(
                f"set reference target success, reference_target_id: {reference_target_id}"
            )
        else:
            # 参考靶标设定响应消息
            send_msg = {
                "cmd": "setreferencetarget",
                "body": {
                    "code": 200,
                    "did": MQTTConfig.getattr("did"),
                    "at": get_now_time(),
                    "reference_target": "",
                    "msg": "set failed, reference_target not exist",
                },
                "msgid": "bb6f3eeb2",
            }
            mqtt_send_queue.put(send_msg)
            logger.warning(
                f"reference target {reference_target} not found in camera0_id2boxstate."
            )

    @staticmethod
    def receive_getstatus_msg(received_msg: dict | None = None):
        """设备状态查询消息"""
        global need_send_get_status_msg
        logger.success("received getstatus msg")
        # 需要发送设备状态查询消息
        need_send_get_status_msg = True

    @staticmethod
    def receive_get_image_msg(received_msg: dict | None = None):
        """现场图像查询消息"""
        logger.info("received upload image msg")
        # {
        #     "cmd":"getimage",
        #     "msgid":"bb6f3eeb2"
        # }
        try:
            image0_timestamp, image0, _ = camera0_queue.get(timeout=get_picture_timeout)
            image1_timestamp, image1, _ = camera1_queue.get(timeout=get_picture_timeout)
            logger.info(
                f"`upload image` get image success, image_timestamp: {image0_timestamp}, {image1_timestamp}"
            )
            # 保存图片
            image_path0 = save_dir / "upload_image0.jpg"
            image_path1 = save_dir / "upload_image1.jpg"
            save_image(image0, image_path0)
            save_image(image1, image_path1)
            logger.info(
                f"save `upload image` success, save image to {image_path0}, {image_path1}"
            )

            # 现场图像查询响应消息
            send_msg = {
                "cmd": "getimage",
                "body": {
                    "code": 200,
                    "did": MQTTConfig.getattr("did"),
                    "at": get_now_time(),
                    "msg": "upload succeed",
                    "path": [str(image_path0), str(image_path1)],  # 图片本地路径
                    "img": ["upload_image0.jpg", "upload_image1.jpg"],  # 文件名称
                },
                "msgid": "bb6f3eeb2",
            }
            mqtt_send_queue.put(send_msg)
            logger.success(
                f"upload image send msg success, image_path: {image_path0}, {image_path1}"
            )
        except queue.Empty:
            logger.warning("upload image send msg failed")
            # 现场图像查询响应消息
            send_msg = {
                "cmd": "setconfig",
                "body": {
                    "code": 200,
                    "did": MQTTConfig.getattr("did"),
                    "at": get_now_time(),
                    "msg": "upload failed, get image timeout",
                    "path": [],  # 图片本地路径
                    "img": [],  # 文件名称
                },
                "msgid": "bb6f3eeb2",
            }
            mqtt_send_queue.put(send_msg)
            get_picture_timeout_process()

    @staticmethod
    def receive_temp_control_msg(received_msg: dict | None = None):
        """温控板回复控温指令"""
        global received_temp_control_msg
        # {
        #     "cmd": "askadjusttempdata",
        #     "times": "2024-09-11T15:45:30",
        #     "camera": "2",
        #     "param": {
        #         "result": "OK/NOT"
        #     },
        #     "msgid": 1
        # }
        logger.success(f"received askadjusttempdata response: {received_msg}")
        received_temp_control_msg = True

    @staticmethod
    def receive_temp_data_msg(received_msg: dict | None = None):
        """日常温度数据"""

        global temperature_data
        global need_send_temp_control_msg
        # {
        #     "cmd": "sendtempdata",
        #     "camera": "2",
        #     "times": "2024-09-11T15:45:30",
        #     "param": {
        #         "inside_air_t": 10,
        #         "exterior_air_t": 10,
        #         "sensor1_t": 10,
        #         "sensor2_t": 10,
        #         "sensor3_t": 10,
        #         "sensor4_t": 257,
        #         "sensor5_t": 257,
        #         "sensor6_t": 257
        #     },
        #     "msgid": 1
        # }
        _temperature_data: dict = received_msg.get("param", {})
        # logger.info(f"received temp data: {_temperature_data}")
        # temperature_data: {
        #     'L3_WK_1': 10,
        #     'L3_WK_2': 10,
        #     'L3_WK_3': 10,
        #     'L3_WK_4': 10,
        #     'L3_WK_5': 10,
        #     'L3_WK_6': 257
        #     'L3_WK_7': 257
        #     'L3_WK_8': 257
        # }
        temperature_data = {
            f"L3_WK_{i+1}": v for i, v in enumerate(_temperature_data.values())
        }
        # logger.info(f"received temp data transform to temperature_data: {temperature_data}")

        # TODO: 临时温度保护措施
        # 只有 inside_air_t 小于控制的温度时才控温
        target_temperature: float = TemperatureConfig.getattr("target_temperature")
        inside_air_t: float = _temperature_data.get("inside_air_t", 100)
        if inside_air_t < target_temperature - 1 and need_send_temp_control_msg:
            logger.success(
                f"inside air temperature `{inside_air_t}` is lower than target temperature `{target_temperature}`, start adjust temperature."
            )
            Send.send_temperature_control_msg(target_temperature, 0)
            Send.send_temperature_control_msg(target_temperature, 1)
            need_send_temp_control_msg = False

    @staticmethod
    def receive_adjust_temp_data_msg(received_msg: dict | None = None):
        """温度调节过程数据"""
        # {
        #     "cmd": "sendadjusttempdata",
        #     "camera": "2",
        #     "times": "2024-09-11T15:45:30",
        #     "param": {
        #         "parctical_t": 10,
        #         "control_t": 10,
        #         "control_way": "warm/cold",
        #         "pwm_data": 10
        #     },
        #     "msgid": 1
        # }
        # logger.info(f"received adjust temp data: {received_msg}")
        Send.send_temperature_change_msg(received_msg.get("param", {}))

    @staticmethod
    def receive_stop_adjust_temp_data(received_msg: dict | None = None):
        """温控停止消息"""
        global is_temp_stable
        global need_send_in_working_state_msg
        # {
        #     "cmd": "stopadjusttemp",
        #     "camera": "2",
        #     "times": "2024-09-11T15:45:30",
        #     "param": {
        #         "current_t": 10,
        #         "control_t": 10
        #     },
        #     "msgid": 1
        # }
        logger.success(
            f"received stop adjust temp data: {received_msg.get('param', {})}"
        )
        is_temp_stable = True
        need_send_in_working_state_msg = True

    @staticmethod
    def receive_reboot_msg(received_msg: dict | None = None):
        """重启终端设备消息"""
        # {
        #     "cmd":"reboot",
        #     "msgid":"bb6f3eeb2",
        # }
        # 重启终端设备响应消息
        send_msg = {
            "cmd": "reboot",
            "body": {
                "did": MQTTConfig.getattr("did"),
                "at": get_now_time(),
                "msg": "reboot succeed",
            },
            "msgid": "e37e42c53",
        }
        mqtt_send_queue.put(send_msg)
        logger.success("reboot device")
        time.sleep(5)
        os._exit()

    @staticmethod
    def receive_update_config_file_msg(received_msg: dict | None = None):
        """更新配置文件消息"""
        logger.info("received update config file msg")
        # {
        #     "cmd":"updateconfigfile",
        #     "body":{
        #         "path":"results/config_20240919-170300.yaml"
        #     },
        #     "msgid": "bb6f3eeb2"
        # }
        try:
            # 旧的温度
            old_target_temperature: float = TemperatureConfig.getattr(
                "target_temperature"
            )

            new_config_path = Path(received_msg["body"]["path"])
            # 载入配置文件
            load_config_from_yaml(config_path=new_config_path)

            # 新的温度
            new_target_temperature: float = TemperatureConfig.getattr(
                "target_temperature"
            )
            if old_target_temperature != new_target_temperature:
                Send.send_temperature_control_msg(new_target_temperature, 0)
                Send.send_temperature_control_msg(new_target_temperature, 1)

            # 更新配置文件响应消息
            send_msg = {
                "cmd": "updateconfigfile",
                "body": {
                    "code": 200,
                    "did": MQTTConfig.getattr("did"),
                    "at": get_now_time(),
                    "msg": "update succeed",
                },
                "msgid": "bb6f3eeb2",
            }
            mqtt_send_queue.put(send_msg)
            logger.success(f"update config file: {new_config_path} success")

        except Exception as e:
            # 更新配置文件响应消息
            send_msg = {
                "cmd": "updateconfigfile",
                "body": {
                    "code": 200,
                    "did": MQTTConfig.getattr("did"),
                    "at": get_now_time(),
                    "msg": "update error",
                },
                "msgid": "bb6f3eeb2",
            }
            mqtt_send_queue.put(send_msg)
            logger.error(f"update config file: {new_config_path} error: {e}")

    @staticmethod
    def receive_get_config_file_msg(received_msg: dict | None = None):
        """查询配置文件消息"""
        logger.info("received send config file msg")
        # {
        #     "cmd":"getconfigfile",
        #     "msgid": "bb6f3eeb2"
        # }
        # 查询配置文件响应消息
        send_msg = {
            "cmd": "getconfigfile",
            "body": {
                "code": 200,
                "did": MQTTConfig.getattr("did"),
                "at": get_now_time(),
                "msg": "get succeed",
                "path": str(save_dir / "config_runtime.yaml"),  # 配置文件本地路径
                "config": "config_runtime.yaml",  # 文件名称
            },
            "msgid": "bb6f3eeb2",
        }
        mqtt_send_queue.put(send_msg)
        logger.success(
            f"get config file success, config_path: {save_dir / 'config_runtime.yaml'}"
        )

    @staticmethod
    def receive_ask_adjust_led_level_with_time_msg(received_msg: dict | None = None):
        """温控板回复补光灯控制命令（带开启时间）"""

        # {
        #     "cmd": "askadjustLEDlevel",
        #     "times": "2024-09-11T15:45:30",
        #     "param":{
        #         "result": "OK/NOT"
        #     },
        #     "msgid": 1
        # }
        if received_msg.get("param", {}).get("result", "NOT") == "OK":
            logger.success("received askadjustLEDlevel response OK")
        else:
            logger.error("received askadjustLEDlevel response NOT OK")

    @staticmethod
    def receive_ask_open_led_level_msg(received_msg: dict | None = None):
        """温控板回复补光灯开启命令"""

        # {
        #     "cmd": "askopenLED",
        #     "times": "2024-09-11 15:45:30",
        #     "param": {
        #         "result": "OK/NOT"
        #     },
        #     "msgid": 1
        # }
        if received_msg.get("param", {}).get("result", "NOT") == "OK":
            logger.success("received askopenLED response OK")
        else:
            logger.error("received askopenLED response NOT OK")

    @staticmethod
    def receive_ask_close_led_msg(received_msg: dict | None = None):
        """温控板回复补光灯关闭命令"""

        # {
        #     "cmd": "askcloseLED",
        #     "times": "2024-09-11 15:45:30",
        #     "param": {
        #         "result": "OK/NOT"
        #     },
        #     "msgid": 1
        # }
        if received_msg.get("param", {}).get("result", "NOT") == "OK":
            logger.success("received askcloseLED response OK")
        else:
            logger.error("received askcloseLED response NOT OK")


class Send:
    @staticmethod
    def main_send_msg():
        # 设备部署响应消息
        if need_send_device_deploying_msg:
            Send.send_device_deploying_msg()

        # 发送靶标校正响应消息
        if need_send_target_correction_msg:
            Send.send_target_correction_msg()

        # 发送删除靶标响应消息
        if need_send_delete_target_msg:
            Send.send_delete_target_msg()

        # 发送添加靶标响应消息
        if need_send_set_target_msg:
            Send.send_set_target_msg()

        # 设备状态查询响应消息
        if need_send_get_status_msg:
            Send.send_getstatus_msg()

        # 设备进入工作状态消息
        # 温度正常
        if need_send_in_working_state_msg:
            Send.send_in_working_state_msg()

        # 温度异常告警消息
        if False:
            Send.send_temperature_alarm_msg()

        # 设备异常告警消息
        if False:
            Send.send_device_error_msg()

        # 设备温控变化消息
        if False:
            Send.send_temperature_change_msg()

        # 温度控制命令
        if False:
            Send.send_temperature_control_msg()

        # 相机超时错误
        get_picture_timeout_threshold: int = MainConfig.getattr(
            "get_picture_timeout_threshold"
        )
        get_picture_timeout_count: int = MainConfig.getattr("get_picture_timeout_count")
        if get_picture_timeout_count >= get_picture_timeout_threshold:
            logger.warning(
                f"get picture timeout count: {get_picture_timeout_count} >= threshold: {get_picture_timeout_threshold}, send device error msg"
            )
            Send.send_device_error_msg("camera timeout")
            MainConfig.setattr("get_picture_timeout_count", 0)

    @staticmethod
    def get_xyz(cycle_results: dict[int, dict]):
        # cycle_results: {
        #     "0": {
        #         "image_timestamp": "image--20240927-105451.193949--0",
        #         "box": [1808, 1034, 1906, 1132],
        #         "center": [1856.2635982759466, 1082.2241236800633],
        #         "radii": [19.452733149311193, 42.37471354702409],
        #         "distance": 3958.5385113630155,
        #         "exposure_time": 140685,
        #         "offset": [0, 0]
        #     },
        #     ...
        # }
        ndigits: int = RingsLocationConfig.getattr("ndigits")
        data = {
            f"L1_SJ_{k+1}": {
                "box": v["box"],
                "X": round(v["center"][0], ndigits),
                "Y": round(v["center"][1], ndigits),
                "Z": round(v["distance"], ndigits),
            }
            for k, v in cycle_results.items()
            if v["center"] is not None
        }
        return data

    @staticmethod
    def send_device_deploying_msg():
        """设备部署响应消息"""
        global need_send_device_deploying_msg

        logger.info("send device deploying msg")

        if not camera0_cycle_results or not camera1_cycle_results:
            logger.warning(
                "camera0_cycle_results or camera1_cycle_results is empty, can't send device deploying msg, wait for next cycle."
            )
            return

        try:
            # 获取照片
            _, image0, _ = camera0_queue.get(timeout=get_picture_timeout)
            _, image1, _ = camera1_queue.get(timeout=get_picture_timeout)
            image_path0 = save_dir / "deploy0.jpg"
            image_path1 = save_dir / "deploy1.jpg"
            save_image(image0, image_path0)
            save_image(image1, image_path1)
            logger.info(
                f"save `deploying image` success, save image to {image_path0}, {image_path1}"
            )
            path = [str(image_path0), str(image_path1)]
            img = ["deploy0.jpg", "deploy1.jpg"]
        except queue.Empty:
            path = []
            img = []
            get_picture_timeout_process()

        main_camera_index: int = CameraConfig.getattr("main_camera_index")
        camera_left_index: int = CameraConfig.getattr("camera_left_index")
        camera0_data = Send.get_xyz(camera0_cycle_results)
        camera1_data = Send.get_xyz(camera1_cycle_results)
        # TODO: 求 Z 轴距离
        _data = (
            deepcopy(camera0_data) if main_camera_index == 0 else deepcopy(camera1_data)
        )
        # 移除 camera0_data 和 camera1_data 中的 z 轴数据
        for v in camera0_data.values():
            v.pop("Z", None)
        for v in camera1_data.values():
            v.pop("Z", None)

        src_data = {
            "left_cam": camera0_data if camera_left_index == 0 else camera1_data,
            "right_cam": camera1_data if camera_left_index == 0 else camera0_data,
        }
        _data["src_data"] = src_data
        logger.info(f"send device deploying data: {_data}")

        send_msg = {
            "cmd": "devicedeploying",
            "body": {
                "code": 200,
                "msg": "deployed succeed",
                "did": MQTTConfig.getattr("did"),
                "type": "deploying",
                "at": get_now_time(),
                "sw_version": "230704180",  # 版本号
                "data": _data,
                # "data": {                              # 靶标初始位置
                #     "L1_SJ_1": {"box": [x1, y1, x2, y2], "X": 9.01, "Y": 8.01, "Z": 2.51},
                #     "L1_SJ_2": {"box": [x1, y1, x2, y2], "X": 4.09, "Y": 8.92, "Z": 4.01},
                #     "src_data": {
                #         "left_cam": {
                #             "L1_SJ_1": {"box": [x1, y1, x2, y2], "X": 9.01, "Y": 8.01},
                #             "L1_SJ_2": {"box": [x1, y1, x2, y2], "X": 4.09, "Y": 8.92},
                #         },
                #         "right_cam": {
                #             "L1_SJ_1": {"box": [x1, y1, x2, y2], "X": 9.01, "Y": 8.01},
                #             "L1_SJ_2": {"box": [x1, y1, x2, y2], "X": 4.09, "Y": 8.92},
                #         }
                #     }
                # }
                "path": path,  # 图片本地路径
                "img": img,  # 文件名称
            },
            "msgid": "bb6f3eeb2",
        }
        mqtt_send_queue.put(send_msg)
        need_send_device_deploying_msg = False
        logger.success("send device deploying msg success")

    @staticmethod
    def send_target_correction_msg():
        """发送靶标校正响应消息"""
        global need_send_target_correction_msg

        logger.info("send target correction msg")

        if not camera0_cycle_results:
            logger.warning(
                "camera0_cycle_results is empty, can't send target correction msg, wait for next cycle."
            )
            return

        _data = Send.get_xyz(camera0_cycle_results)

        # 靶标校正响应消息
        send_msg = {
            "cmd": "targetcorrection",
            "body": {
                "code": 200,
                "did": MQTTConfig.getattr("did"),
                "at": get_now_time(),
                "msg": "correction succeed",
                "data": _data,
                # "data": { # 靶标初始位置
                #     "L1_SJ_1": {"box": [x1, y1, x2, y2], "X": 19.01, "Y": 18.31, "Z": 10.8},
                #     "L1_SJ_2": {"box": [x1, y1, x2, y2], "X": 4.09, "Y": 8.92, "Z": 6.7},
                #     "L1_SJ_3": {"box": [x1, y1, x2, y2], "X": 2.02, "Y": 5.09, "Z": 14.6}
                # },
            },
            "msgid": "bb6f3eeb2",
        }
        mqtt_send_queue.put(send_msg)
        need_send_target_correction_msg = False
        logger.success("target correction success")

    @staticmethod
    def send_delete_target_msg():
        """发送删除靶标响应消息"""
        global need_send_delete_target_msg

        logger.info("send delete target msg")

        if not camera0_cycle_results:
            logger.warning(
                "camera0_cycle_results is mepty, can't send delete target msg, wait for next cycle."
            )
            return

        try:
            # 获取照片
            _, image0, _ = camera0_queue.get(timeout=get_picture_timeout)
            image_path = save_dir / "delete_target.jpg"
            save_image(image0, image_path)
            logger.info(f"save `delete target` success, save image to {image_path}")
            path = str(image_path)
            img = "delete_target.jpg"
        except queue.Empty:
            path = ""
            img = ""
            get_picture_timeout_process()

        _data = Send.get_xyz(camera0_cycle_results)

        # 删除靶标响应消息
        send_msg = {
            "cmd": "deletetarget",
            "body": {
                "code": 200,
                "did": MQTTConfig.getattr("did"),
                "at": get_now_time(),
                "msg": "delete target succeed",
                "data": _data,
                # "data": { # 靶标初始位置
                #     "L1_SJ_1": {"box": [x1, y1, x2, y2], "X": 19.01, "Y": 18.31, "Z": 10.8},
                #     "L1_SJ_2": {"box": [x1, y1, x2, y2], "X": 4.09, "Y": 8.92, "Z": 6.7},
                #     "L1_SJ_3": {"box": [x1, y1, x2, y2], "X": 2.02, "Y": 5.09, "Z": 14.6}
                # },
                "path": path,  # 图片本地路径
                "img": img,  # 文件名称
            },
            "msgid": "bb6f3eeb2",
        }
        mqtt_send_queue.put(send_msg)
        need_send_delete_target_msg = False
        logger.success("send delete target msg success")

    @staticmethod
    def send_set_target_msg():
        """发送添加靶标响应消息"""
        global need_send_set_target_msg

        logger.info("send set target msg")

        if not camera0_cycle_results:
            logger.warning(
                "camera0_cycle_results is empty, can't send set target msg, wait for next cycle."
            )
            return

        try:
            # 获取照片
            _, image0, _ = camera0_queue.get(timeout=get_picture_timeout)
            image_path = save_dir / "set_target.jpg"
            save_image(image0, image_path)
            logger.info(f"save `set target` success, save image to {image_path}")
            path = str(image_path)
            img = "set_target.jpg"
        except queue.Empty:
            path = ""
            img = ""
            get_picture_timeout_process()

        _data = Send.get_xyz(camera0_cycle_results)

        # 添加靶标响应消息
        send_msg = {
            "cmd": "settarget",
            "body": {
                "code": 200,
                "did": MQTTConfig.getattr("did"),
                "at": get_now_time(),
                "msg": "set target succeed",
                "data": _data,
                # "data": { # 靶标初始位置
                #     "L1_SJ_1": {"box": [x1, y1, x2, y2], "X": 19.01, "Y": 18.31, "Z": 10.8},
                #     "L1_SJ_2": {"box": [x1, y1, x2, y2], "X": 4.09, "Y": 8.92, "Z": 6.7},
                #     "L1_SJ_3": {"box": [x1, y1, x2, y2], "X": 2.02, "Y": 5.09, "Z": 14.6}
                # },
                "path": path,  # 图片本地路径
                "img": img,  # 文件名称
            },
            "msgid": "bb6f3eeb2",
        }
        mqtt_send_queue.put(send_msg)
        need_send_set_target_msg = False
        logger.success("send set target msg success")

    @staticmethod
    def send_getstatus_msg():
        """设备状态查询响应消息"""
        global need_send_get_status_msg

        logger.info("send getstatus msg")
        # camera0_cycle_results: {
        #     "0": {
        #         "image_timestamp": "image--20240927-105451.193949--0",
        #         "box": [1808, 1034, 1906, 1132],
        #         "center": [1856.2635982759466, 1082.2241236800633],
        #         "radii": [19.452733149311193, 42.37471354702409],
        #         "distance": 3958.5385113630155,
        #         "exposure_time": 140685,
        #         "offset": [0, 0]
        #     },
        #     ...
        # }

        # sensor_state = {
        #     "L3_WK_1": 0,  # 0表示无错误，-1供电异常，
        #     "L3_WK_2": 0,  # -2传感器数据异常，-3采样间隔内没有采集到数据
        #     "L3_WK_3": 0,
        #     "L3_WK_4": 0,
        #     "L3_WK_5": 0,
        #     "L3_WK_6": -1,
        # }
        sensor_state = {k: 0 for k, v in temperature_data.items()}

        camera0_standard_results: dict | None = RingsLocationConfig.getattr(
            "camera0_standard_results"
        )
        camera1_standard_results: dict | None = RingsLocationConfig.getattr(
            "camera1_standard_results"
        )
        box_sensor_state = {}
        if camera0_standard_results and camera0_cycle_results:
            for k in camera0_standard_results.keys():
                if k in camera0_cycle_results.keys():
                    v = camera0_cycle_results[k]
                    if v["box"] is not None and v["center"] is not None:
                        box_sensor_state[f"L1_SJ_{k+1}"] = 0
                    else:
                        box_sensor_state[f"L1_SJ_{k+1}"] = -2
                else:
                    box_sensor_state[f"L1_SJ_{k+1}"] = -2
        else:
            logger.warning(
                "camera0_cycle_results or camera0_standard_results is None, wait for next cycle"
            )
            return

        if camera1_standard_results and camera1_cycle_results:
            for k in camera1_standard_results.keys():
                if k in camera1_cycle_results.keys():
                    v = camera1_cycle_results[k]
                    if v["box"] is not None and v["center"] is not None:
                        box_sensor_state[f"L1_SJ_{k+1}"] = 0
                    else:
                        box_sensor_state[f"L1_SJ_{k+1}"] = -2
                else:
                    box_sensor_state[f"L1_SJ_{k+1}"] = -2
        else:
            logger.warning(
                "camera1_cycle_results or camera1_standard_results is None, wait for next cycle"
            )
            return

        sensor_state.update(box_sensor_state)

        send_msg = {
            "cmd": "getstatus",
            "body": {
                "ext_power_volt": 38.3,  # 供电电压
                "temp": 20,  # 环境温度
                "signal_4g": -84.0,  # 4g信号强度
                "sw_version": "230704180",  # 固件版本号
                "sensor_state": sensor_state,
                # "sensor_state": {
                #     "L3_WK_1": 0,# 0表示无错误，-1供电异常，
                #     "L3_WK_2": 0,# -2传感器数据异常，-3采样间隔内没有采集到数据
                #     "L3_WK_3": 0,
                #     "L3_WK_4": 0,
                #     "L3_WK_5": 0,
                #     "L3_WK_6": -1,
                #     "L1_SJ_0": 0,
                #     "L1_SJ_1": 0,
                #     "L1_SJ_2": 0,
                # }
            },
            "msgid": "bb6f3eeb2",
        }
        mqtt_send_queue.put(send_msg)
        need_send_get_status_msg = False
        logger.success("send getstatus msg success")

    @staticmethod
    def send_in_working_state_msg():
        global need_send_in_working_state_msg
        """设备进入工作状态响应消息"""
        send_msg = {
            "cmd": "devicestate",
            "body": {
                "did": MQTTConfig.getattr("did"),
                "type": "working",
                "at": get_now_time(),
                "code": 200,
                "msg": "device working",
            },
        }
        mqtt_send_queue.put(send_msg)
        logger.success("send working state msg")
        need_send_in_working_state_msg = False

    @staticmethod
    def send_temperature_alarm_msg():
        """温度异常告警消息"""
        send_msg = {
            "cmd": "alarm",
            "body": {
                "did": MQTTConfig.getattr("did"),
                "type": "temperature",
                "at": get_now_time(),
                "number": [1, 3, 4],
                "data": {
                    "L3_WK_1": 80,
                    "L3_WK_2": 20,
                    "L3_WK_3": 80,
                    "L3_WK_4": 90,
                    "L3_WK_5": 20,
                    "L3_WK_6": 20,
                },
            },
        }
        mqtt_send_queue.put(send_msg)
        logger.warning("send temperature alarm msg")

    @staticmethod
    def send_device_error_msg(msg: str = "device error"):
        """设备异常告警消息"""
        send_msg = {
            "cmd": "alarm",
            "body": {
                "did": MQTTConfig.getattr("did"),
                "type": "device",
                "at": get_now_time(),
                "code": 400,
                "msg": msg,
            },
        }
        mqtt_send_queue.put(send_msg)
        logger.warning("send device error msg")

    @staticmethod
    def send_temperature_change_msg(data: dict):
        """温控变化消息"""
        # data: {
        #     "parctical_t": 10,
        #     "control_t": 10,
        #     "control_way": "warm/cold",
        #     "pwm_data": 10
        # }
        # logger.info(f"send temperature change data: {data}")
        send_msg = {
            "cmd": "devicestate",
            "body": {
                "did": MQTTConfig.getattr("did"),
                "type": "temperature_control",
                "at": get_now_time(),
                "data": {
                    "L3_WK_1": data.get("parctical_t", 0),  # 内仓实际温度
                    "control_t": data.get("control_t", 0),  # 目标温度
                    "control_way": data.get("control_t", "warm"),
                    "pwm_data": data.get("pwm_data", 10),
                },
            },
        }
        mqtt_send_queue.put(send_msg)
        logger.success("send temperature change msg success")

    @staticmethod
    def send_temperature_control_msg(
        temperature: int = 25,
        camera: int = 0,
    ):
        """温度控制命令"""
        global received_temp_control_msg
        global is_temp_stable

        logger.info(
            f"send temperature control, temperature: {temperature}, camera: {camera}"
        )
        # {
        #     "cmd":"adjusttempdata",
        #     "param":{
        #         "control_t":10，
        #         "camera":"2"
        #     },
        #     "msgid":1
        # }
        send_msg = {
            "cmd": "adjusttempdata",
            "param": {"control_t": temperature, "camera": f"{camera}"},
            "msgid": 1,
        }
        serial_send_queue.put(send_msg)
        logger.success("send temperature control msg success")
        received_temp_control_msg = False
        is_temp_stable = False

    @staticmethod
    def send_adjust_led_level_with_time_msg(adjust_led_level_param: dict):
        """补光灯控制命令（带开启时间）"""
        logger.info(f"send adjust led level with time: {adjust_led_level_param}")
        max_led_level: int = AdjustCameraConfig.getattr("max_led_level")
        if adjust_led_level_param["level"] < 1:
            adjust_led_level_param["level"] = 1
            logger.warning("adjust led level less than 1, set to 1")
        if adjust_led_level_param["level"] > max_led_level:
            adjust_led_level_param["level"] = max_led_level
            logger.warning(f"adjust led level greater than {max_led_level}, set to {max_led_level}")
        # {
        #     "cmd": "adjustLEDlevel",
        #     "param": {
        #         "level": 5,
        #         "times": 1
        #     },
        #     "msgid": 1
        # }
        send_msg = {
            "cmd": "adjustLEDlevel",
            "param": adjust_led_level_param,
            "msgid": 1,
        }
        serial_send_queue.put(send_msg)
        logger.success("send adjust led level with time msg success")

    @staticmethod
    def send_open_led_level_msg(led_level: int = 1):
        """补光灯开启命令"""
        logger.info(f"send open led level : {led_level}")
        max_led_level: int = AdjustCameraConfig.getattr("max_led_level")
        if led_level < 1:
            led_level = 1
            logger.warning("adjust led level less than 1, set to 1")
        if led_level > max_led_level:
            led_level = max_led_level
            logger.warning(f"adjust led level greater than {max_led_level}, set to {max_led_level}")
        # {
        #     "cmd": "openLED",
        #     "param": {
        #         "level": 5,
        #     },
        #     "msgid": 1
        # }
        send_msg = {
            "cmd": "openLED",
            "param": {
                "level": led_level,
            },
            "msgid": 1,
        }
        serial_send_queue.put(send_msg)
        logger.success("send open led level msg success")

    @staticmethod
    def send_close_led_msg():
        """补光灯关闭命令"""
        logger.info("send close led msg")
        send_msg = {
            "cmd": "closeLED",
            "param": {
                "reserve": 257, #保留参数，暂时没用，赋值为257
            },
            "msgid": 1
        }
        serial_send_queue.put(send_msg)
        logger.success("send close led level msg success")


if __name__ == "__main__":
    main()
