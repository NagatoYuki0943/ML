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
from utils import clear_queue, save_to_jsonl, load_standard_cycle_results, drop_excessive_queue_items


# 将日志输出到文件
# 每天 0 点新创建一个 log 文件
handler_id = logger.add('log/runtime_{time}.log', rotation='00:00')


def main() -> None:
    #------------------------------ 初始化 ------------------------------#
    logger.info("init start")

    save_dir: Path = MainConfig.getattr("save_dir")
    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")

    #-------------------- 基础 --------------------#
    # 主线程消息队列
    main_queue = queue.Queue()

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

    logger.success("init end")
    #------------------------------ 初始化 ------------------------------#

    #------------------------------ 调整曝光 ------------------------------#
    try:
        _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
        cv2.imwrite(save_dir / "image_default.jpg", image)
    except queue.Empty:
        logger.error("get picture timeout")

    logger.info("ajust exposure 1 start")
    adjust_exposure_full_res_for_loop(camera_queue)
    logger.success("ajust exposure 1 end")
    try:
        _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
        cv2.imwrite(save_dir / "image_adjust_exposure.jpg", image)
    except queue.Empty:
        logger.error("get picture timeout")
    #------------------------------ 调整曝光 ------------------------------#

    #-------------------- 循环变量 --------------------#
    # 主循环
    i = 0
    # 一个周期内的结果
    cycle_results = {}
    # 初始的坐标
    standard_cycle_results = None
    logger.info(f"standard_cycle_results: {standard_cycle_results}")
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
                adjust_exposure_full_res_for_loop(camera_queue)
                logger.info("full image ajust exposure end")
                #-------------------- 调整全图曝光 --------------------#


                #-------------------- 设定循环 --------------------##
                # 总的循环轮数为 1 + 曝光次数
                total_cycle_loop_count = 2
                logger.critical(f"During this cycle, there will be {total_cycle_loop_count} iters.")
                # 当前周期，采用从 0 开始
                cycle_loop_count = 0
                logger.info(f"The {cycle_loop_count} iter within the cycle.")
                #-------------------- 设定循环 --------------------##

                # 周期设置
                cycle_before_time = cycle_current_time

            else:
                #-------------------- camera capture --------------------#
                logger.info(f"The {cycle_loop_count + 1} iter within the cycle.")

                # 忽略多于图像
                drop_excessive_queue_items(camera_queue)

                try:
                    # 获取照片
                    image_timestamp, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
                    logger.info(f"camera get image: {image_timestamp}, ExposureTime = {image_metadata['ExposureTime']}, AnalogueGain = {image_metadata['AnalogueGain']}, shape = {image.shape}")
            #-------------------- camera capture --------------------#

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
                        #------------------------- 整理检测结果 -------------------------#

                        #------------------------- 结束周期 -------------------------#
                        cycle_results = {} # 重置周期内结果
                        cycle_loop_count = -1   # 重置周期内循环计数
                        logger.success(f"The cycle is over.")
                        #------------------------- 结束周期 -------------------------#

        # 检测周期外
        if cycle_loop_count == -1:
            ...

        # 主循环休眠
        main_sleep_interval: int = MainConfig.getattr("main_sleep_interval")
        time.sleep(main_sleep_interval / 1000)

        logger.warning(f"{i = }")
        i += 1


if __name__ == "__main__":
    main()
