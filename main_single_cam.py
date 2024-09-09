import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import json
from threading import Lock, Thread
import queue
from loguru import logger
from typing import Literal
from pathlib import Path

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
)

from camera_engine import camera_engine
from find_target import find_target, find_around_target, find_lost_target
from adjust_camera import adjust_exposure2, adjust_exposure3
from serial_communication import serial_receive, serial_send
from mqtt_communication import mqtt_receive, mqtt_send
from utils import clear_queue


# 将日志输出到文件
# 每天 0 点新创建一个 log 文件
handler_id = logger.add('log/runtime_{time}.log', rotation='00:00')


def main() -> None:
    #------------------------------ 初始化 ------------------------------#
    logger.info("init start")

    #-------------------- 基础 --------------------#
    # 主线程消息队列
    main_queue = queue.Queue()

    save_dir: Path = MainConfig.getattr("save_dir")
    location_save_dir = MainConfig.getattr("location_save_dir")
    camera_result_save_path = MainConfig.getattr("camera_result_save_path")
    get_picture_timeout = MainConfig.getattr("get_picture_timeout")
    #-------------------- 基础 --------------------#

    #-------------------- 运行时配置 --------------------#

    #-------------------- 运行时配置 --------------------#

    #-------------------- 历史 --------------------#

    #-------------------- 历史 --------------------#

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

    logger.success("init end")
    #------------------------------ 初始化 ------------------------------#

    #------------------------------ 调整曝光 ------------------------------#
    try:
        _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
        cv2.imwrite(save_dir / "image_default.jpg", image)
    except queue.Empty:
        logger.error("get picture timeout")

    logger.info("ajust exposure 1 start")
    adjust_exposure3(camera_queue)
    logger.success("ajust exposure 1 end")
    try:
        _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
        cv2.imwrite(save_dir / "image_adjust_exposure.jpg", image)
    except queue.Empty:
        logger.error("get picture timeout")
    #------------------------------ 调整曝光 ------------------------------#

    #------------------------------ 找到目标 ------------------------------#
    logger.info("find target start")
    image_timestamp: str
    image: np.ndarray
    image_metadata: dict

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
        id2boxstate, got_target_number = find_target(rectified_image)
        logger.info(f"image find target id2boxstate: \n{id2boxstate}")
        logger.info(f"image find target number: {got_target_number}")
        if got_target_number < target_number:
            # 数量不够，发送告警
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

    #------------------------------ 找到目标 ------------------------------#

    #-------------------- 循环变量 --------------------#
    # 主循环
    i = 0
    # 一个周期内的结果
    cycle_results = {}
    # 初始的坐标
    init_cycle_results = None
    # 一个周期内总循环次数
    total_cycle_loop_count = 0
    # 一个周期内循环计数
    cycle_loop_count = -1
    # 每个周期的间隔时间
    cycle_time_interval: int = MainConfig.getattr("cycle_time_interval")
    cycle_before_time = time.time()

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
                adjust_exposure3(camera_queue)
                logger.info("full image ajust exposure end")
                #-------------------- 调整全图曝光 --------------------#

                #-------------------- 找到目标 --------------------#
                try:
                    _, image, _ = camera_queue.get(timeout=get_picture_timeout)
                    #-------------------- 畸变矫正 --------------------#
                    rectified_image = image
                    #-------------------- 畸变矫正 --------------------#

                    #-------------------- 小区域模板匹配 --------------------#
                    _, got_target_number = find_around_target(rectified_image)
                    if got_target_number == 0:
                        logger.warning("no target found in the image")
                        continue
                    #-------------------- 小区域模板匹配 --------------------#
                except queue.Empty:
                    logger.error("get picture timeout")
                #-------------------- 找到目标 --------------------#

                #-------------------- 调整 box 曝光 --------------------#
                logger.info("boxes ajust exposure start")
                id2boxstate = MatchTemplateConfig.getattr("id2boxstate")
                # 每次开始调整曝光
                # ex: {72000: array([[1327, 1697, 1828, 2198]]), 78000: array([[1781,  811, 2100, 1130]])}
                exposure2id2boxstate = adjust_exposure3(camera_queue, id2boxstate)
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
                            _box = boxestate['box']
                            x1, y1, x2, y2 = _box
                            target = rectified_image[y1:y2, x1:x2]

                            try:
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
                                logger.success(f"{result = }")
                                logger.success(f"box {j} rings location success")
                                center = np.array([result['center_x_mean'] + _box[0], result['center_y_mean'] + _box[1]])
                                cycle_results[j] = {
                                    'image_timestamp': f"image--{image_timestamp}--{j}",
                                    'box': _box,
                                    'center': center,
                                    'exposure_time': exposure_time,
                                }
                                # 保存到文件
                                string = json.dumps(result, ensure_ascii=False)
                                with open(camera_result_save_path, mode='a', encoding='utf-8') as f:
                                    f.write(string + "\n")
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
                            # 结束周期
                            #------------------------- 检查是否丢失目标 -------------------------#
                            _target_number = MatchTemplateConfig.getattr("target_number")
                            _got_target_number = MatchTemplateConfig.getattr("got_target_number")
                            if _target_number > _got_target_number:
                                logger.warning(f"The target number {_target_number} is not enough, got {_got_target_number} targets, start to find lost target.")
                                # 丢失目标
                                try:
                                    _, image, _ = camera_queue.get(timeout=get_picture_timeout)
                                    #-------------------- 畸变矫正 --------------------#
                                    rectified_image = image
                                    #-------------------- 畸变矫正 --------------------#

                                    #-------------------- 模板匹配 --------------------#
                                    find_lost_target(rectified_image)
                                    #-------------------- 模板匹配 --------------------#

                                    _target_number = MatchTemplateConfig.getattr("target_number")
                                    _got_target_number = MatchTemplateConfig.getattr("got_target_number")
                                    if _target_number > _got_target_number:
                                        # 重新查找完成之后仍然不够, 发送告警
                                        logger.critical(f"The target number {_target_number} is not enough, got {_got_target_number} targets.")
                                    else:
                                        # 丢失目标重新找回
                                        logger.success(f"The lost target has been found, the target number {_target_number} is enough, got {_got_target_number} targets.")

                                except queue.Empty:
                                    logger.error("get picture timeout")
                            else:
                                # 目标数量正常
                                logger.success(f"The target number {_target_number} is enough, got {_got_target_number} targets.")
                            #------------------------- 检查是否丢失目标 -------------------------#

                            #------------------------- 整理检测结果 -------------------------#
                            logger.success(f"{cycle_results = }")

                            # [n, 2] n个目标中心坐标
                            # init_cycle_centers: {0: array([1830.03952661, 1097.25685946]), 1: array([2090.1380529 , 2148.12593385])}
                            # new_cycle_centers: {0: array([1830.05961465, 1097.2564746 ]), 1: array([2090.13342415, 2148.12260239])}

                            # 初始化 init_cycle_centers
                            if init_cycle_results is None:
                                new_cycle_centers = {k: result['center'] for k, result in cycle_results.items()}
                                if any(v is None for v in new_cycle_centers.values()):
                                    logger.warning("Some box not found in new_cycle_centers, can't init init_cycle_centers.")
                                else:
                                    init_cycle_results = cycle_results
                                    logger.info(f"init init_cycle_results: {init_cycle_results}")
                            else:
                                move_threshold = RingsLocationConfig.getattr("move_threshold")
                                init_cycle_centers = {k: result['center'] for k, result in init_cycle_results.items()}
                                new_cycle_centers = {k: result['center'] for k, result in cycle_results.items()}
                                # 计算移动距离
                                for l in init_cycle_results.keys():
                                    if l in new_cycle_centers.keys() and new_cycle_centers[l] is not None:
                                        for n in range(2): # 0 1 代表 x y
                                            move_distance = abs(init_cycle_centers[l][n] - new_cycle_centers[l][n])
                                            if move_distance > move_threshold:
                                                logger.warning(f"box {l} {'x' if n == 0 else 'y'} move distance {move_distance} is over threshold {move_threshold}.")
                                            else:
                                                logger.info(f"box {l} {'x' if n == 0 else 'y'} move distance {move_distance} is under threshold {move_threshold}.")
                                    else:
                                        logger.warning(f"box {l} not found in cycle_centers.")

                            #------------------------- 整理检测结果 -------------------------#

                            #------------------------- 结束周期 -------------------------#
                            cycle_results = {} # 重置周期内结果
                            cycle_loop_count = -1   # 重置周期内循环计数
                            logger.success(f"The cycle is over.")
                            #------------------------- 结束周期 -------------------------#
                        else:
                            # 不是结束周期，设置下一轮的曝光值
                            exposure_time = cycle_exposure_times[cycle_loop_count]
                            CameraConfig.setattr("exposure_time", exposure_time)

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
