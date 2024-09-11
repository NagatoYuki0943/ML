import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import cv2
import json
from threading import Lock, Thread
import queue
from loguru import logger
from typing import Literal
from pathlib import Path
import sys

from algorithm import (
    ThreadWrapper,
    adaptive_threshold_rings_location,
    StereoCalibration,
    RaspberryMQTT,
    RaspberrySerialPort,
)
from config import (
    MainConfig,
    StereoCalibrationConfig,
    MatchTemplateConfig,
    CameraConfig,
    RingsLocationConfig,
    AdjustCameraConfig,
    MQTTConfig,
    SerialCommConfig,
    ALL_CONFIGS,
    init_config_from_yaml,
    load_config_from_yaml,
    save_config_to_yaml,
)

from camera_engine import camera_engine
from find_target import find_target, find_around_target, find_lost_target
from adjust_camera import (
    adjust_exposure_full_res_for_loop,
    adjust_exposure_low_res_for_loop, # 调整分辨率需要一段时间才能获取调整后的图片分辨率
)
from serial_communication import serial_receive, serial_send
from mqtt_communication import mqtt_receive, mqtt_send
from utils import clear_queue, save_to_jsonl, load_standard_cycle_results, get_now_time, save_image


# 将日志输出到文件
# 每天 0 点新创建一个 log 文件
handler_id = logger.add('log/runtime_{time}.log', rotation='00:00')


def main() -> None:
    #------------------------------ 初始化 ------------------------------#
    logger.info("init start")

    save_dir: Path = MainConfig.getattr("save_dir")
    location_save_dir: Path = MainConfig.getattr("location_save_dir")
    camera_result_save_path: Path = MainConfig.getattr("camera_result_save_path")
    history_save_path: Path = MainConfig.getattr("history_save_path")
    standard_save_path: Path = MainConfig.getattr("standard_save_path")
    original_config_path: Path = MainConfig.getattr("original_config_path")                   # 默认 config, 用于重置
    runtime_config_path: Path = MainConfig.getattr("runtime_config_path")   # 运行时 config, 用于临时修改配置
    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")

    # 保存原始配置
    save_config_to_yaml(config_path=original_config_path)
    # 从运行时 config 加载配置
    init_config_from_yaml(config_path=runtime_config_path)
    logger.success("init config success")

    #-------------------- 基础 --------------------#
    # 主线程消息队列
    main_queue = queue.Queue()
    image_timestamp: str
    image: np.ndarray
    image_metadata: dict

    #-------------------- 基础 --------------------#

    #-------------------- 运行时配置 --------------------#

    #-------------------- 运行时配置 --------------------#

    #-------------------- 初始化相机 --------------------#
    logger.info("开始初始化相机")
    camera0_thread = ThreadWrapper(
        target_func = camera_engine,
        queue_maxsize = CameraConfig.getattr("queue_maxsize"),
        camera_index = 0,
    )
    camera0_thread.start()
    camera_queue = camera0_thread.queue

    time.sleep(1)
    logger.success("初始化相机完成")
    #-------------------- 初始化相机 --------------------#

    #-------------------- 畸变矫正 --------------------#
    logger.info("开始初始化畸变矫正")
    stereo_calibration = StereoCalibration(
        StereoCalibrationConfig.getattr('camera_matrix_left'),
        StereoCalibrationConfig.getattr('camera_matrix_right'),
        StereoCalibrationConfig.getattr('distortion_coefficients_left'),
        StereoCalibrationConfig.getattr('distortion_coefficients_right'),
        StereoCalibrationConfig.getattr('R'),
        StereoCalibrationConfig.getattr('T'),
        StereoCalibrationConfig.getattr('pixel_width_mm'),
    )
    logger.success("初始化畸变矫正完成")
    #-------------------- 畸变矫正 --------------------#

    #-------------------- 初始化串口 --------------------#
    # logger.info("开始初始化串口")
    # serial_comm = RaspberrySerialPort(
    #     SerialCommConfig.temperature_logger,
    #     SerialCommConfig.port,
    #     SerialCommConfig.baudrate,
    #     SerialCommConfig.timeout,
    #     SerialCommConfig.BUFFER_SIZE,
    # )
    # serial_send_thread = ThreadWrapper(
    #     target_func = serial_send,
    #     ser = serial_comm,
    # )
    # serial_receive_thread = Thread(
    #     target = serial_receive,
    #     kwargs={'ser':serial_comm, 'queue':main_queue}
    # )
    # serial_send_queue = serial_send_thread.queue
    # serial_receive_thread.start()
    # serial_send_thread.start()
    # logger.success("初始化串口完成")
    #-------------------- 初始化串口 --------------------#

    #-------------------- 初始化MQTT客户端 --------------------#
    # logger.info("开始初始化MQTT客户端")
    # mqtt_comm = RaspberryMQTT(
    #     MQTTConfig.broker,
    #     MQTTConfig.port,
    #     MQTTConfig.timeout,
    #     MQTTConfig.topic,
    # )
    # mqtt_receive_thread = Thread(
    #     target = mqtt_receive,
    #     kwargs={'client':mqtt_comm, 'queue':main_queue},
    #     daemon=True
    # )
    # mqtt_send_thread = ThreadWrapper(
    #     target_func = mqtt_send,
    #     client = mqtt_comm,
    # )
    # mqtt_send_queue = mqtt_send_thread.queue
    # mqtt_receive_thread.start()
    # mqtt_send_thread.start()
    # logger.success("初始化MQTT客户端完成")
    #-------------------- 初始化MQTT客户端 --------------------#

    # 设备启动消息
    send_msg = {
        "cmd":"devicestate",
        "body":{
            "did": "458796",
            "type": "startup",
            "at": get_now_time(),
            "sw_version": "230704180", # 版本号
            "code": 200,
            "msg": "device starting"
        }
    }

    logger.success("init end")
    #------------------------------ 初始化 ------------------------------#

    #------------------------------ 调整曝光 ------------------------------#
    try:
        _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
        save_image(image, save_dir / "image_default.jpg")
    except queue.Empty:
        logger.error("get picture timeout")

    logger.info("ajust exposure 1 start")
    adjust_exposure_full_res_for_loop(camera_queue)
    logger.success("ajust exposure 1 end")
    try:
        _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
        save_image(image, save_dir / "image_adjust_exposure.jpg")
    except queue.Empty:
        logger.error("get picture timeout")
    #------------------------------ 调整曝光 ------------------------------#

    #------------------------------ 找到目标 ------------------------------#
    logger.info("find target start")
    #-------------------- 取图 --------------------#
    try:
        _, image, _ = camera_queue.get(timeout=get_picture_timeout)
        logger.info(f"{image.shape = }, ExposureTime = {image_metadata['ExposureTime']}, AnalogueGain = {image_metadata['AnalogueGain']}")
        #-------------------- 取图 --------------------#

        #-------------------- 畸变矫正 --------------------#
        logger.info("rectify image start")
        rectified_image = image
        logger.success("rectify image success")
        #-------------------- 畸变矫正 --------------------#

        #-------------------- 模板匹配 --------------------#
        logger.info("image find target start")

        target_number: int = MatchTemplateConfig.getattr("target_number")
        id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr("id2boxstate")
        # {
        #     i: {
        #         "ratio": ratio,
        #         "score": score,
        #         "box": box
        #     }
        # }
        if id2boxstate is None:
            logger.warning("id2boxstate is None, use find_target instead of find_around_target")
            id2boxstate, got_target_number = find_target(rectified_image)
            # 初始化时将 target_number 设置为找到的 target 数量
            MatchTemplateConfig.setattr("target_number", got_target_number)
            logger.success(f"update target_number to {got_target_number}")
        else:
            logger.success("id2boxstate is not None, use find_around_target")
            id2boxstate, got_target_number = find_around_target(rectified_image)
        logger.info(f"image find target id2boxstate: \n{id2boxstate}")
        logger.info(f"image find target number: {got_target_number}")
        if got_target_number < target_number:
            # 数量不够，发送告警
            ...
            if got_target_number == 0:
                # 没找到任何目标，发送告警
                ...

        boxes = [boxestate['box'] for boxestate in id2boxstate.values()]
        # 绘制boxes
        image_draw = image.copy()
        # image_draw = undistorted_image.copy()
        for i in range(len(boxes)):
            cv2.rectangle(
                img = image_draw,
                pt1 = (boxes[i][0], boxes[i][1]),
                pt2 = (boxes[i][2], boxes[i][3]),
                color = (255, 0, 0),
                thickness = 3
            )
        plt.figure(figsize=(10, 10))
        plt.imshow(image_draw, cmap='gray')
        plt.savefig(save_dir / "image_match_template.png")
        plt.close()
        logger.success("image find target success")
        #-------------------- 模板匹配 --------------------#

        logger.success("find target end")
    except queue.Empty:
        logger.error("get picture timeout")

    # 保存运行时配置
    save_config_to_yaml(config_path=runtime_config_path)
    #------------------------------ 找到目标 ------------------------------#

    #-------------------- 循环变量 --------------------#
    # 主循环
    i = 0
    # 一个周期内的结果
    cycle_results = {}
    # 初始的坐标
    standard_cycle_results = load_standard_cycle_results(standard_save_path)
    logger.info(f"standard_cycle_results: {standard_cycle_results}")
    # 一个周期内总循环次数
    total_cycle_loop_count = 0
    # 一个周期内循环计数
    cycle_loop_count = -1
    # 每个周期的间隔时间
    cycle_time_interval: int = MainConfig.getattr("cycle_time_interval")
    cycle_before_time = time.time()

    # 是否需要发送部署信息
    need_send_devicedeploying_msg = False
    # 是否收到温控回复命令
    received_temp_control_msg = True
    # 温度是否平稳
    is_temp_stable = False

    #-------------------- 循环变量 --------------------#

    while True:
        cycle_current_time = time.time()
        # 取整为时间周期
        _cycle_before_time_period = int(cycle_before_time * 1000 // cycle_time_interval)
        _cycle_current_time_period = int(cycle_current_time * 1000 // cycle_time_interval)
        # 进入周期
        # 条件为 当前时间周期大于等于前一个时间周期 或者 周期已经开始运行
        if _cycle_current_time_period > _cycle_before_time_period or cycle_loop_count > -1:
            if cycle_loop_count == -1:  # 每个周期的第一次循环
                logger.success(f"The cycle is started.")
                #-------------------- 调整全图曝光 --------------------#
                logger.info("full image ajust exposure start")
                adjust_exposure_full_res_for_loop(camera_queue)
                logger.info("full image ajust exposure end")
                #-------------------- 调整全图曝光 --------------------#

                #-------------------- 补光灯 --------------------#
                if False:
                    # 补光灯控制命令
                    send_msg = {
                        "cmd":"adjustLEDlevel",
                        "param":{
                            "level":10,
                            "times":1
                        },
                        "msgid":1
                    }

                    # 回复补光灯调节指令
                    {
                        "cmd":" askadjustLEDlevel ",
                        "times": "2024-09-11T15:45:30",
                        "param": {
                            "result": "OK/NOT"
                        },
                        "msgid": 1
                    }
                #-------------------- 补光灯 --------------------#

                #-------------------- 找到目标 --------------------#
                try:
                    _, image, _ = camera_queue.get(timeout=get_picture_timeout)
                    #-------------------- 畸变矫正 --------------------#
                    rectified_image = image
                    #-------------------- 畸变矫正 --------------------#

                    #-------------------- 小区域模板匹配 --------------------#
                    _, got_target_number = find_around_target(rectified_image)
                    if got_target_number == 0:
                        # ⚠️⚠️⚠️ 本次循环没有找到目标，尝试寻找遗失的目标 ⚠️⚠️⚠️
                        logger.warning("no target found in the image, start find_lost_target")
                        _, got_target_number = find_lost_target(rectified_image)
                        if got_target_number == 0:
                            logger.error("no target found in the image, exit")
                        continue
                    #-------------------- 小区域模板匹配 --------------------#
                except queue.Empty:
                    logger.error("get picture timeout")
                #-------------------- 找到目标 --------------------#

                #-------------------- 调整 box 曝光 --------------------#
                logger.info("boxes ajust exposure start")
                id2boxstate: dict[int, dict] | None  = MatchTemplateConfig.getattr("id2boxstate")
                # 每次开始调整曝光
                # ex: {72000: array([[1327, 1697, 1828, 2198]]), 78000: array([[1781,  811, 2100, 1130]])}
                exposure2id2boxstate = adjust_exposure_full_res_for_loop(camera_queue, id2boxstate)
                cycle_exposure_times = list(exposure2id2boxstate.keys())
                logger.info(f"exposure2boxes: {exposure2id2boxstate}")
                logger.info("boxes ajust exposure end")
                #-------------------- 调整 box 曝光 --------------------#

                #-------------------- 设定循环 --------------------##
                # 总的循环轮数为 1 + 曝光次数
                total_cycle_loop_count = 1 + len(exposure2id2boxstate)
                logger.critical(f"During this cycle, there will be {total_cycle_loop_count} iters.")
                # 当前周期，采用从 0 开始
                cycle_loop_count = 0
                logger.info(f"The {cycle_loop_count} iter within the cycle.")

                # 设置下一轮的曝光值
                exposure_time = cycle_exposure_times[cycle_loop_count]
                CameraConfig.setattr("exposure_time", exposure_time)
                #-------------------- 设定循环 --------------------##

                # 周期设置
                cycle_before_time = cycle_current_time

            else:
                #-------------------- camera capture --------------------#
                camera_qsize = camera_queue.qsize()
                if camera_qsize > 0:
                    logger.info(f"The {cycle_loop_count + 1} iter within the cycle.")

                    # 忽略多余的图片
                    if camera_qsize > 1:
                        logger.warning(f"camera got {camera_qsize} frames, ignore {camera_qsize - 1} frames")
                        for _ in range(camera_qsize - 1):
                            try:
                                camera_queue.get(timeout=get_picture_timeout)
                            except queue.Empty:
                                logger.error("get picture timeout")

                    try:
                        # 获取照片
                        image_timestamp, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
                        logger.info(f"camera get image: {image_timestamp}, ExposureTime = {image_metadata['ExposureTime']}, AnalogueGain = {image_metadata['AnalogueGain']}, shape = {image.shape}")
                #-------------------- camera capture --------------------#

                        #------------------------- 检测目标 -------------------------#
                        # 获取检测参数
                        gradient_threshold_percent: float = RingsLocationConfig.getattr("gradient_threshold_percent")
                        iters: int = RingsLocationConfig.getattr("iters")
                        order: int = RingsLocationConfig.getattr("order")
                        rings_nums: int = RingsLocationConfig.getattr("rings_nums")
                        min_group_size: int = RingsLocationConfig.getattr("min_group_size")
                        sigmas: int = RingsLocationConfig.getattr("sigmas")
                        draw_scale: int = RingsLocationConfig.getattr("draw_scale")
                        save_grads: bool = RingsLocationConfig.getattr("save_grads")
                        save_detect_images: bool = RingsLocationConfig.getattr("save_detect_images")
                        save_detect_results: bool = RingsLocationConfig.getattr("save_detect_results")

                        #-------------------- 畸变矫正 --------------------#
                        rectified_image = image
                        #-------------------- 畸变矫正 --------------------#

                        #-------------------- single box location --------------------#
                        # for循环，截取图像
                        exposure_time = cycle_exposure_times[cycle_loop_count]
                        id2boxstate = exposure2id2boxstate[exposure_time]
                        for j, boxestate in id2boxstate.items():
                            _box: list | None = boxestate['box']
                            try:
                                # box 可能为 None, 使用 try except 处理
                                x1, y1, x2, y2 = _box
                                target = rectified_image[y1:y2, x1:x2]

                                logger.info(f"box {j} rings location start")
                                result = adaptive_threshold_rings_location(
                                    target,
                                    f"image--{image_timestamp}--{j}",
                                    iters,
                                    order,
                                    rings_nums,
                                    min_group_size,
                                    sigmas,
                                    location_save_dir,
                                    draw_scale,
                                    save_grads,
                                    save_detect_images,
                                    save_detect_results,
                                    gradient_threshold_percent,
                                )
                                result['metadata'] = image_metadata
                                # 保存到文件
                                save_to_jsonl(result, camera_result_save_path)
                                logger.success(f"{result = }")
                                logger.success(f"box {j} rings location success")
                                center = [float(result['center_x_mean'] + _box[0]), float(result['center_y_mean'] + _box[1])]
                                cycle_results[j] = {
                                    'image_timestamp': f"image--{image_timestamp}--{j}",
                                    'box': _box,
                                    'center': center,
                                    'exposure_time': exposure_time,
                                }
                            except Exception as e:
                                logger.error(e)
                                logger.error(f"box {j} rings location failed")
                                cycle_results[j] = {
                                    'image_timestamp': f"image--{image_timestamp}--{j}",
                                    'box': _box,
                                    'center': None, # 丢失目标, 置为 None
                                    'exposure_time': exposure_time,
                                }
                        #-------------------- single box location --------------------#

                        #------------------------- 检测目标 -------------------------#

                    except queue.Empty:
                        logger.error("get picture timeout")

                    else:
                        # 没有发生错误
                        # 周期内循环计数加1
                        cycle_loop_count += 1

                        # 正常判断是否结束周期
                        if cycle_loop_count == total_cycle_loop_count - 1:
                            #------------------------- 整理检测结果 -------------------------#
                            logger.success(f"{cycle_results = }")
                            # 保存到文件
                            save_to_jsonl(cycle_results, history_save_path)

                            # [n, 2] n个目标中心坐标
                            # init_cycle_centers: {0: [1830.03952661, 1097.25685946]), 1: [2090.1380529 , 2148.12593385]}
                            # new_cycle_centers: {0: [1830.05961465, 1097.2564746 ]), 1: [2090.13342415, 2148.12260239]}

                            # 初始化 init_cycle_centers
                            if standard_cycle_results is None:
                                new_cycle_centers = {k: result['center'] for k, result in cycle_results.items()}
                                if any(v is None for v in new_cycle_centers.values()):
                                    logger.warning("Some box not found in new_cycle_centers, can't init init_cycle_centers.")
                                else:
                                    standard_cycle_results = cycle_results
                                    save_to_jsonl(standard_cycle_results, standard_save_path, mode='w')
                                    logger.info(f"init standard_cycle_results: {standard_cycle_results}")
                            else:
                                move_threshold = RingsLocationConfig.getattr("move_threshold")
                                init_cycle_centers = {k: result['center'] for k, result in standard_cycle_results.items()}
                                new_cycle_centers = {k: result['center'] for k, result in cycle_results.items()}

                                # 超出距离的 box id
                                distance_result = {}
                                over_distance_ids = set()
                                # 计算移动距离
                                for l in standard_cycle_results.keys():
                                    if l in new_cycle_centers.keys() and new_cycle_centers[l] is not None:
                                        # for n in range(2): # 0 1 代表 x y
                                        #     move_distance = abs(init_cycle_centers[l][n] - new_cycle_centers[l][n])
                                        #     if move_distance > move_threshold:
                                        #         over_distance_ids.add(l)
                                        #         logger.warning(f"box {l} {'x' if n == 0 else 'y'} move distance {move_distance} is over threshold {move_threshold}.")
                                        #     else:
                                        #         logger.info(f"box {l} {'x' if n == 0 else 'y'} move distance {move_distance} is under threshold {move_threshold}.")
                                        x_move_distance = abs(init_cycle_centers[l][0] - new_cycle_centers[l][0])
                                        y_move_distance = abs(init_cycle_centers[l][1] - new_cycle_centers[l][1])
                                        distance_result[l] = (x_move_distance, y_move_distance)
                                        if x_move_distance > move_threshold:
                                            over_distance_ids.add(l)
                                            logger.warning(f"box {l} x move distance {x_move_distance} is over threshold {move_threshold}.")
                                        else:
                                            logger.info(f"box {l} x move distance {x_move_distance} is under threshold {move_threshold}.")
                                        if y_move_distance > move_threshold:
                                            over_distance_ids.add(l)
                                            logger.warning(f"box {l} y move distance {y_move_distance} is over threshold {move_threshold}.")
                                        else:
                                            logger.info(f"box {l} y move distance {y_move_distance} is under threshold {move_threshold}.")
                                    else:
                                        # box没找到将移动距离设置为 1e6
                                        distance_result[l] = (1e6, 1e6)
                                        over_distance_ids.add(l)
                                        logger.warning(f"box {l} not found in cycle_centers.")

                                logger.info(f"distance_result: {distance_result}")
                                logger.info(f"over_distance_ids: {over_distance_ids}")
                                send_msg_data = {f"L1_SJ_{k}": {'X': v[0], 'Y': v[1]} for k, v in distance_result.items()}
                                logger.info(f"send_msg_data: {send_msg_data}")
                                if len(over_distance_ids) > 0:
                                    # ⚠️⚠️⚠️ 有box移动距离超过阈值 ⚠️⚠️⚠️
                                    logger.warning(f"box {over_distance_ids} move distance is over threshold {move_threshold}.")

                                    # 保存丢失的图片
                                    image_path = save_dir / f"target_displacement.jpg"
                                    save_image(image, image_path)
                                    # 位移告警消息
                                    send_msg = {
                                        "cmd": "alarm",
                                        "body": {
                                            "did": "458796",
                                            "type": "displacement",
                                            "at": get_now_time(),
                                            "number": [1],# 表示异常的靶标编号
                                            "data": send_msg_data,
                                            "ftpurl": "/5654/20240810160846",# ftp上传路径
                                            "img": [image_path]# 文件名称
                                        }
                                    }
                                else:
                                    # ✅️✅️✅️ 所有 box 移动距离都小于阈值 ✅️✅️✅️
                                    logger.success(f"All box move distance is under threshold {move_threshold}.")
                                    # 正常数据消息
                                    send_msg = {
                                        "cmd": "update",
                                        "did": "458796",
                                        "data": send_msg_data
                                    }

                            #------------------------- 整理检测结果 -------------------------#

                            #------------------------- 检查是否丢失目标 -------------------------#
                            target_number = MatchTemplateConfig.getattr("target_number")
                            got_target_number = MatchTemplateConfig.getattr("got_target_number")

                            # 丢失目标
                            if target_number > got_target_number:
                                logger.warning(f"The target number {target_number} is not enough, got {got_target_number} targets, start to find lost target.")

                                try:
                                    _, image, _ = camera_queue.get(timeout=get_picture_timeout)
                                    #-------------------- 畸变矫正 --------------------#
                                    rectified_image = image
                                    #-------------------- 畸变矫正 --------------------#

                                    #-------------------- 模板匹配 --------------------#
                                    _, got_target_number = find_lost_target(rectified_image)
                                    #-------------------- 模板匹配 --------------------#

                                    if target_number > got_target_number:
                                        # ❌️❌️❌️ 重新查找完成之后仍然不够 ❌️❌️❌️
                                        # 获取丢失的box id
                                        id2boxstate: dict[int, dict] | None  = MatchTemplateConfig.getattr("id2boxstate")
                                        loss_ids = [i for i, boxestate in id2boxstate.items() if boxestate['box'] is None]

                                        logger.critical(f"The target number {target_number} is not enough, got {got_target_number} targets, loss box ids: {loss_ids}.")

                                        # 保存丢失的图片
                                        image_path = save_dir / f"target_loss.jpg"
                                        save_image(image, image_path)
                                        send_msg = {
                                            "cmd":"alarm",
                                            "body":{
                                                "did": "458796",
                                                "type": "target_loss",
                                                "at": get_now_time(),
                                                "number": loss_ids,# 异常的靶标编号
                                                "data": send_msg_data,
                                                "ftpurl": "/5654/20240810160846",# ftp上传路径
                                                "img": [image_path]# 文件名称
                                            }
                                        }
                                    else:
                                        # ✅️✅️✅️ 丢失目标重新找回 ✅️✅️✅️
                                        logger.success(f"The lost target has been found, the target number {target_number} is enough, got {got_target_number} targets.")

                                except queue.Empty:
                                    logger.error("get picture timeout")

                            # 目标数量正常
                            else:
                                logger.success(f"The target number {target_number} is enough, got {got_target_number} targets.")
                            #------------------------- 检查是否丢失目标 -------------------------#

                            #------------------------- 结束周期 -------------------------#
                            cycle_results = {} # 重置周期内结果
                            cycle_loop_count = -1   # 重置周期内循环计数
                            logger.success(f"The cycle is over.")
                            #------------------------- 结束周期 -------------------------#
                        else:
                            # 不是结束周期，设置下一轮的曝光值
                            exposure_time = cycle_exposure_times[cycle_loop_count]
                            CameraConfig.setattr("exposure_time", exposure_time)

        # 检测周期外
        if cycle_loop_count == -1:
            # 需要发送设备部署消息
            if need_send_devicedeploying_msg:
                {
                    "cmd":"devicedeploying",
                    "result":"succ/fail",
                    "body":{
                        "code":200,
                        "msg":"deployed succeed",
                        "did":"458796",
                        "type":"deploying",
                        "at":"2023-08-09T16:08:46Z",
                        "sw_version":"230704180", # 版本号
                        "code":200,
                        "msg":"device starting",
                        "data": { # 靶标初始位置
                            "L1_SJ_1":{"X":19.01,"Y":18.31,"Z":10.8},
                            "L1_SJ_2":{"X":4.09,"Y":8.92,"Z":6.7},
                            "L1_SJ_3":{"X":2.02,"Y":5.09,"Z":14.6}
                        },
                        "ftpurl":"/5654/20240810160846",
                        "img":["1.jpg","2.jpg"]
                    },
                    "msgid": "bb6f3eeb2"
                }
                ...
                need_send_devicedeploying_msg = False

            #------------------------- 获取消息 -------------------------#
            while not main_queue.empty():
                received_msg = main_queue.get()
                cmd = received_msg.get('cmd')
                logger.info(f"received msg: {received_msg}")

                # 设备部署消息
                if cmd == 'devicedeploying':
                    # {
                    #     "cmd":"devicedeploying",
                    #     "msgid":"bb6f3eeb2",
                    # }
                    logger.info("device deploying, reset config and init target")
                    # 设备部署，重置配置和初始靶标
                    load_config_from_yaml(config_path=original_config_path)
                    standard_cycle_results = None
                    logger.success("reset config and init target success")
                    # 发送部署消息
                    need_send_devicedeploying_msg = True

                # 靶标校正消息
                elif cmd == 'targetcorrection':
                    # {
                    #     "cmd":"targetcorrection",
                    #     "msgid":"bb6f3eeb2",
                    #     "body":{
                    #         "add_boxes":[
                    #             [x1, y1, x2, y2],
                    #             [x1, y1, x2, y2],
                    #         ],
                    #         "remove_box_ids":["L1_SJ_3","L1_SJ_4"]
                    #     }
                    # }
                    logger.info("target correction, update target")
                    # 靶标丢失，传递来新靶标
                    # id2boxstate: {
                    #     i: {
                    #         "ratio": ratio,
                    #         "score": score,
                    #         "box": box
                    #     }
                    # }
                    id2boxstate: dict[int, dict] | None = MatchTemplateConfig.getattr("id2boxstate")
                    remove_box_ids: dict[int, list] = received_msg['body']['remove_box_ids']
                    # 去除多余的 box
                    for remove_box_id in remove_box_ids:
                        id2boxstate.pop(int(remove_box_id.split('_')[-1]))

                    # 将新的 box 转换为列表
                    new_boxstates = [{"ratio": None, "score": None, "box": box} for box in received_msg['body']['add_boxes']]
                    # 旧的 box 也转换为列表，并合并
                    old_boxstates = list(id2boxstate.values())
                    old_boxstates.extend(new_boxstates)
                    # 合并后的 box 生成新的 id2boxstate
                    id2boxstate = {i: boxstate for i, boxstate in enumerate(old_boxstates)}

                    # 设置新目标数量和靶标信息
                    MatchTemplateConfig.setattr("target_number", len(id2boxstate))
                    MatchTemplateConfig.setattr("id2boxstate", id2boxstate)

                    # 因为重设了靶标，所以需要重新初始化标准靶标
                    standard_cycle_results = None

                    # send_msg = {
                    #     "cmd":"targetcorrection",
                    #     "body":{
                    #         "code":200,
                    #         "msg":"correction succeed"
                    #         "data": {
                    #             "L1_SJ_1":{"X":19.01,"Y":18.31,"Z":10.8},
                    #             "L1_SJ_2":{"X":4.09,"Y":8.92,"Z":6.7},
                    #             "L1_SJ_3":{"X":2.02,"Y":5.09,"Z":14.6}
                    #         },
                    #     },
                    #     "msgid": "bb6f3eeb2"
                    # }
                    # 发送更新靶标消息
                    raise NotImplementedError("target correction send msg not implemented")
                    logger.success(f"update target success, new id2boxstate: {id2boxstate}")

                # 参考靶标设定消息
                elif cmd == 'setreferencetarget':
                    logger.info("set reference target")
                    # {
                    #     "cmd":"setreferencetarget",
                    #     "msgid":"bb6f3eeb2",
                    #     "apikey":"e343f59e9a1b426aa435",
                    #     "body":{
                    #         "reference_target":"L1_SJ_1"
                    #     }
                    # }
                    reference_target = received_msg['body']['reference_target']
                    reference_target_id = int(reference_target.split('_')[-1])
                    MatchTemplateConfig.setattr("reference_target_ids", [reference_target_id])
                    send_msg = {
                        "cmd": "setreferencetarget",
                        "result": "succ/fail",
                        "body": {
                            "code": 200,
                            "msg": "set succeed"
                        },
                        "msgid": "bb6f3eeb2"
                    }
                    raise NotImplementedError("set reference target send msg not implemented")
                    logger.success(f"set reference target success, reference_target_id: {reference_target_id}")

                # 设备状态查询消息
                elif cmd == 'getstatus':
                    send_msg = {
                        "cmd": "getstatus",
                        "body": {
                            "ext_power_volt": 38.3,# 供电电压
                            "temp": 20,# 环境温度
                            "signal_4g": -84.0,# 4g信号强度
                            "sw_version": "230704180",# 固件版本号
                            "sensor_state":{
                                "sensor1": 0,# 0表示无错误，-1供电异常，
                                "sensor2": 0,# -2传感器数据异常，-3采样间隔内没有采集到数据
                                "sensor3": 0,
                                "sensor4": 0,
                                "sensor5": 0,
                                "sensor6": 0, # sensor1~6为温度传感器，其余为靶标
                                "sensor7": 0,
                                "sensor8": 0,
                                "sensor9": 0,
                            }
                        },
                        "msgid": "bb6f3eeb2"
                    }
                    raise NotImplementedError("get status send msg not implemented")

                # 现场图像查询消息
                elif cmd == 'getimage':
                    logger.warning(f"get image")
                    # {
                    #     "cmd":"getimage",
                    #     "msgid":"bb6f3eeb2"
                    # }
                    try:
                        image_timestamp, image, _ = camera_queue.get(timeout=get_picture_timeout)
                        # 保存图片
                        image_path = save_dir / f"upload_image.jpg"
                        save_image(image, image_path)

                        send_msg = {
                            "cmd": "setconfig",
                            "result": "succ",
                            "body":{
                                "code": 200,
                                "msg": "upload succeed",
                                "ftpurl": "/5654/20240810160846",# ftp上传路径
                                "img": [image_path]# 文件名称
                            },
                            "msgid": "bb6f3eeb2"
                        }

                        logger.success(f"get image success, image_path: {image_path}")
                    except queue.Empty:
                        send_msg = {
                            "cmd":"setconfig",
                            "result":"succ",
                            "body":{
                                "code":200,
                                "msg":"upload succeed",
                                "ftpurl":"/5654/20240810160846",# ftp上传路径
                                "img":[]# 文件名称
                            },
                            "msgid": "bb6f3eeb2"
                        }
                        logger.error("get picture timeout")
                    raise NotImplementedError("get image send msg not implemented")

                # 温控板回复控温指令, 回复可能延期
                elif cmd == 'askadjusttempdata':
                    {
                        "cmd": "askadjusttempdata",
                        "times": "2024-09-11T15:45:30",
                        "camera": "2",
                        "param": {
                            "result": "OK/NOT"
                        },
                        "msgid": 1
                    }
                    logger.info("received askadjusttempdata response")
                    received_temp_control_msg = True

                # 日常温度数据
                elif cmd =='sendtempdata':
                    {
                        "cmd":"sendtempdata",
                        "camera":"2",
                        "times":"2024-09-11T15:45:30",
                        "param":{
                            "inside_air_t":10,
                            "exterior_air_t":10,
                            "sensor1_t":10,
                            "sensor2_t":10,
                            "sensor3_t":10,
                            "sensor4_t":257,
                            "sensor5_t":257,
                            "sensor6_t":257
                        },
                        "msgid":1
                    }
                    logger.info("received temp data")

                # 温度调节过程数据
                elif cmd == 'sendadjusttempdata':
                    {
                        "cmd":"sendadjusttempdata",
                        "camera":"2",
                        "times":"2024-09-11T15:45:30",
                        "param":{
                            "parctical_t":10,
                            "control_t":10,
                            "control_way":"warm/cold",
                            "pwm_data":10
                        },
                        "msgid":1
                    }
                    logger.info("received just temp data")

                # 温控停止消息
                elif cmd == 'stopadjusttemp':
                    logger.info("received stop adjust temp data")
                    {
                        "cmd":"stopadjusttemp",
                        "camera":"2",
                        "times":"2024-09-11T15:45:30",
                        "param":{
                            "current_t":10,
                            "control_t":10
                        },
                        "msgid":1
                    }
                    is_temp_stable = True
                    logger.success("received stop adjust temp data")

                # 重启终端设备消息
                elif cmd =='reboot':
                    {
                        "cmd":"reboot",
                        "msgid":"bb6f3eeb2",
                    }
                    # 重启终端设备响应消息
                    send_msg ={
                        "cmd":"reboot",
                        "body":{
                            "code":200,
                            "did":"7804d2",
                            "msg":"reboot succeed",
                        },
                        "msgid": "bb6f3eeb2"
                    }
                    sys.exit()
                else:
                    logger.warning(f"unknown cmd: {cmd}")
                    logger.warning(f"unknown msg: {received_msg}")
                #------------------------- 获取消息 -------------------------#

                #------------------------- 发送消息 -------------------------#
                # 温度异常告警消息
                if False:
                    send_msg = {
                        "cmd": "alarm",
                        "body": {
                            "did": "458796",
                            "type": "temperature",
                            "at": get_now_time(),
                            "number": [1, 3, 4],
                            "data": {
                                "L3_WK_1": 80,
                                "L3_WK_2": 20,
                                "L3_WK_3": 80,
                                "L3_WK_4": 90,
                                "L3_WK_5": 20,
                                "L3_WK_6": 20
                            }
                        }
                    }
                    raise NotImplementedError("send msg not implemented")

                # 设备异常告警消息
                if False:
                    send_msg = {
                        "cmd": "alarm",
                        "body": {
                            "did": "458796",
                            "type": "device",
                            "at": get_now_time(),
                            "code": 400,
                            "msg": "device error"
                        }
                    }
                    raise NotImplementedError("send msg not implemented")

                # 设备进入工作状态消息
                # 温度正常
                if is_temp_stable:
                    send_msg = {
                        "cmd": "devicestate",
                        "body": {
                            "did": "458796",
                            "type": "working",
                            "at": get_now_time(),
                            "code": 200,
                            "msg": "device working"
                        }
                    }

                # 温控变化消息
                if False:
                    send_msg = {
                        "cmd":"devicestate",
                        "body":{
                            "did": "458796",
                            "type": "temperature_control",
                            "at": get_now_time(),
                            "data": {
                                "L3_WK_1": 20, # 内仓实际温度
                                "control_t": 30, # 目标温度
                                "control_way": "warm/cold",
                                "pwm_data": 10
                            }
                        }
                    }

                # 温度控制命令
                if False:
                    send_msg = {
                        "cmd": "adjusttempdata",
                        "param": {
                            "control_t": 10
                        },
                        "msgid": 1
                    }
                    received_temp_control_msg = False

                #------------------------- 发送消息 -------------------------#


            # 保存运行时配置
            save_config_to_yaml(config_path=runtime_config_path)

        # 主循环休眠
        main_sleep_interval: int = MainConfig.getattr("main_sleep_interval")
        time.sleep(main_sleep_interval / 1000)

        # 测试调整相机
        if i > 5000:
            break
        logger.warning(f"{i = }")
        i += 1


if __name__ == "__main__":
    main()
