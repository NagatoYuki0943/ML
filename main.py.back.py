import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import json
from threading import Lock, Thread
from queue import Queue
from loguru import logger
from typing import Literal
from pathlib import Path

from algorithm import (
    ThreadWrapper,
    adaptive_threshold_rings_location,
    RaspberryMQTT,
    RaspberrySerialPort,
)
from config import (
    MainConfig,
    MatchTemplateConfig,
    CameraConfig,
    RingsLocationConfig,
    AdjustCameraConfig,
    MQTTConfig,
    SerialCommConfig,
)

from camera_engine import camera_engine
from find_target import find_target
from adjust_camera import adjust_exposure_by_mean
from serial_communication import serial_receive, serial_send
from mqtt_communication import mqtt_receive, mqtt_send


def main() -> None:
    #------------------------------ 初始化 ------------------------------#
    logger.info("init start")

    # 主线程消息队列
    main_queue = Queue()

    save_dir: Path = MainConfig.getattr("save_dir")
    save_dir.mkdir(parents=True, exist_ok=True)
    location_save_dir = save_dir / "rings_location"
    location_save_dir.mkdir(parents=True, exist_ok=True)
    camera0_result_save_path = save_dir / "camera0_result.jsonl"
    camera1_result_save_path = save_dir / "camera1_result.jsonl"

    #-------------------- 初始化相机 --------------------#
    camera0_thread = ThreadWrapper(
        target_func = camera_engine,
        queue_maxsize = CameraConfig.getattr("queue_maxsize"),
        camera_index = 0,
    )
    camera1_thread = ThreadWrapper(
        target_func = camera_engine,
        queue_maxsize = CameraConfig.getattr("queue_maxsize"),
        camera_index = 1,
    )
    camera0_thread.start()
    camera1_thread.start()
    camera0_thread_queue = camera0_thread.queue
    camera1_thread_queue = camera1_thread.queue

    time.sleep(1)
    logger.success("初始化相机完成")
    #-------------------- 初始化相机 --------------------#

    #-------------------- 畸变矫正 --------------------#

    #-------------------- 畸变矫正 --------------------#

    #-------------------- 初始化串口 --------------------#
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
    # mqtt_comm = RaspberryMQTT(
    #     MQTTConfig.broker,
    #     MQTTConfig.port,
    #     MQTTConfig.timeout,
    #     MQTTConfig.topic,
    # )
    # mqtt_receive_thread = Thread(
    #     target = mqtt_receive,
    #     kwargs={'client':mqtt_comm, 'queue':main_queue},
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


    #------------------------------ 找到目标 ------------------------------#
    logger.info("find target start")
    camera0_timestamp: str
    camera1_timestamp: str
    camera0_image: np.ndarray
    camera1_image: np.ndarray
    camera0_metadata: dict
    camera1_metadata: dict

    #-------------------- 取图 --------------------#
    camera0_timestamp, camera0_image, camera0_metadata = camera0_thread_queue.get()
    camera0_image = cv2.cvtColor(camera0_image, cv2.COLOR_RGB2GRAY)
    logger.warning(f"{camera0_image.shape = }, ExposureTime = {camera0_metadata['ExposureTime']}, AnalogueGain = {camera0_metadata['AnalogueGain']}")
    cv2.imwrite(save_dir / "camera0_match_template_standard.jpg", camera0_image)

    camera1_timestamp, camera1_image, camera1_metadata = camera1_thread_queue.get()
    camera1_image = cv2.cvtColor(camera1_image, cv2.COLOR_RGB2GRAY)
    logger.warning(f"{camera1_image.shape = }, ExposureTime = {camera1_metadata['ExposureTime']}, AnalogueGain = {camera1_metadata['AnalogueGain']}")
    cv2.imwrite(save_dir / "camera1_match_template_standard.jpg", camera1_image)
    #-------------------- 取图 --------------------#

    #-------------------- 畸变矫正 --------------------#
    logger.info("rectify image start")
    logger.success("rectify image success")
    #-------------------- 畸变矫正 --------------------#

    #-------------------- camera0 --------------------#
    logger.info("camera0 find target start")
    camera0_boxes = find_target(camera0_image)
    logger.info(f"camera0 find target result: \n{camera0_boxes}")

    camera0_up_box = camera0_boxes[0]
    camera0_down_box = camera0_boxes[1]

    # 绘制boxes
    camera0_image_draw = camera0_image.copy()
    for i in range(2):
        cv2.rectangle(
            img = camera0_image_draw,
            pt1 = (camera0_boxes[i][0], camera0_boxes[i][1]),
            pt2 = (camera0_boxes[i][2], camera0_boxes[i][3]),
            color = (255, 0, 0),
            thickness = 3
        )
    plt.figure(figsize=(10, 10))
    plt.imshow(camera0_image_draw, cmap='gray')
    plt.savefig(save_dir / "camera0_match_template_standard_target.png")
    plt.close()
    logger.success("camera0 find target success")
    #-------------------- camera0 --------------------#

    #-------------------- camera1 --------------------#
    logger.info("camera1 find target start")
    camera1_boxes = find_target(camera1_image)
    logger.info(f"camera1 find target result: \n{camera1_boxes}")

    camera1_up_box = camera1_boxes[0]
    camera1_down_box = camera1_boxes[1]

    # 绘制boxes
    camera1_image_draw = camera1_image.copy()
    for i in range(2):
        cv2.rectangle(
            img = camera1_image_draw,
            pt1 = (camera1_boxes[i][0], camera1_boxes[i][1]),
            pt2 = (camera1_boxes[i][2], camera1_boxes[i][3]),
            color = (255, 0, 0),
            thickness = 3
        )
    plt.figure(figsize=(10, 10))
    plt.imshow(camera1_image_draw, cmap='gray')
    plt.savefig(save_dir / "camera1_match_template_standard_target.png")
    plt.close()
    logger.success("camera1 find target success")
    #-------------------- camera1 --------------------#


    logger.success("find target end")
    #------------------------------ 找到目标 ------------------------------#

    #------------------------------ 调整曝光 ------------------------------#
    logger.info("adjust exposure start")
    default_capture_mode: str = CameraConfig.getattr("capture_mode")
    default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
    default_return_image_time_interval: int = CameraConfig.getattr("return_image_time_interval")
    # 调整相机配置，加快拍照
    CameraConfig.setattr("capture_mode", AdjustCameraConfig.getattr("capture_mode"))
    CameraConfig.setattr("capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval"))
    CameraConfig.setattr("return_image_time_interval", AdjustCameraConfig.getattr("return_image_time_interval"))

    # 忽略所有的相机拍照
    while not camera0_thread_queue.empty():
        camera0_thread_queue.get()

    while True:
        _, camera0_image, camera0_metadata = camera0_thread_queue.get()
        camera0_image = cv2.cvtColor(camera0_image, cv2.COLOR_RGB2GRAY)
        # camera0_up_target = image[up_box[1]//2:up_box[3]//2, up_box[0]//2:up_box[2]//2]
        camera0_down_target = camera0_image[camera0_down_box[1]//2:camera0_down_box[3]//2, camera0_down_box[0]//2:camera0_down_box[2]//2]
        new_exposure_time, is_ok = adjust_exposure_by_mean(
            camera0_down_target,
            camera0_metadata['ExposureTime'],
            AdjustCameraConfig.getattr("mean_light_suitable_range"),
            AdjustCameraConfig.getattr("adjust_exposure_time_step"),
        )

        CameraConfig.setattr("exposure_time", new_exposure_time)
        logger.info(f"{new_exposure_time = }, {is_ok = }")
        if is_ok == "ok":
            break

    logger.success(f"set {new_exposure_time = }")

    # 还原相机配置
    CameraConfig.setattr("capture_mode", default_capture_mode)
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)

    cv2.imwrite(save_dir / "camera0_adjust_exposure_finish_down_target.jpg", camera0_down_target)
    cv2.imwrite(save_dir / "camera0_adjust_exposure_finish.jpg", camera0_image)
    logger.success("adjust exposure end")

    # 调整曝光后，清空队列，清除调整曝光前的图片
    time.sleep(1)
    while not camera0_thread_queue.empty():
        camera0_thread_queue.get()
    while not camera1_thread_queue.empty():
        camera1_thread_queue.get()
    #------------------------------ 调整曝光 ------------------------------#

    # 主循环
    i = 0

    # 用来保存不同周期的多个camera的检测中心（多个相机拍照可能不同步）
    camera0_up_center = None
    camera0_down_center = None
    camera1_up_center = None
    camera1_down_center = None
    while True:
        #------------------------- 检测目标 -------------------------#
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

        #-------------------- camera0 --------------------#
        camera0_qsize = camera0_thread_queue.qsize()
        if camera0_qsize > 0:
            # 忽略多余的图片
            if camera0_qsize > 1:
                logger.warning(f"camera 0 got {camera0_qsize} frames, ignore {camera0_qsize - 1} frames")
                for _ in range(camera0_qsize - 1):
                    camera0_thread_queue.get()

            # 获取照片
            camera0_timestamp, camera0_image, camera0_metadata = camera0_thread_queue.get()
            logger.info(f"camera 0 get image: {camera0_timestamp} {camera0_image.shape = }, ExposureTime = {camera0_metadata['ExposureTime']}, AnalogueGain = {camera0_metadata['AnalogueGain']}")

            # 畸变矫正

            # 截取图像
            camera0_up_target = camera0_image[camera0_up_box[1]:camera0_up_box[3], camera0_up_box[0]:camera0_up_box[2]]
            camera0_down_target = camera0_image[camera0_down_box[1]:camera0_down_box[3], camera0_down_box[0]:camera0_down_box[2]]
            # cv2.imwrite(location_save_dir / f"camera0_{camera0_timestamp}--up_target.jpg", camera0_up_target)
            # cv2.imwrite(location_save_dir / f"camera0_{camera0_timestamp}--down_target.jpg", camera0_down_target)

            try:
                logger.info(f"camera 0 up rings location start")
                camera0_up_result = adaptive_threshold_rings_location(
                    camera0_up_target,
                    f"camera0--{camera0_timestamp}--up",
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
                camera0_up_result['metadata'] = camera0_metadata
                logger.success(f"{camera0_up_result = }")
                logger.success(f"camera 0 up rings location success")
                camera0_up_center: list[float] = [camera0_up_result['center_x_mean'], camera0_up_result['center_y_mean']]

                string = json.dumps(camera0_up_result, ensure_ascii=False)
                with open(camera0_result_save_path, mode='a', encoding='utf-8') as f:
                    f.write(string + "\n")

            except Exception as e:
                camera0_up_center = None
                logger.error(e)
                logger.error(f"camera 0 up circle location failed")

            try:
                logger.info(f"camera 0 down rings location start")
                camera0_down_result = adaptive_threshold_rings_location(
                    camera0_down_target,
                    f"camera0--{camera0_timestamp}--down",
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
                camera0_down_result['metadata'] = camera0_metadata
                logger.success(f"{camera0_down_result = }")
                logger.success(f"camera 0 down rings location success")
                camera0_down_center: list[float] = [camera0_down_result['center_x_mean'], camera0_down_result['center_y_mean']]

                string = json.dumps(camera0_down_result, ensure_ascii=False)
                with open(camera0_result_save_path, mode='a', encoding='utf-8') as f:
                    f.write(string + "\n")
            except Exception as e:
                camera0_down_center = None
                logger.error(e)
                logger.error(f"camera 0 down circle location failed")
        #-------------------- camera0 --------------------#

        #-------------------- camera1 --------------------#
        camera1_qsize = camera1_thread_queue.qsize()
        if camera1_qsize > 0:
            # 忽略多余的图片
            if camera1_qsize > 1:
                logger.warning(f"camera 1 got {camera1_qsize} frames, ignore {camera1_qsize - 1} frames")
                for _ in range(camera1_qsize - 1):
                    camera1_thread_queue.get()

            # 获取照片
            camera1_timestamp, camera1_image, camera1_metadata = camera1_thread_queue.get()
            logger.info(f"camera 1 get image: {camera1_timestamp} {camera1_image.shape = }, ExposureTime = {camera1_metadata['ExposureTime']}, AnalogueGain = {camera1_metadata['AnalogueGain']}")

            # 畸变矫正

            # 截取图像
            camera1_up_target = camera1_image[camera1_up_box[1]:camera1_up_box[3], camera1_up_box[0]:camera1_up_box[2]]
            camera1_down_target = camera1_image[camera1_down_box[1]:camera1_down_box[3], camera1_down_box[0]:camera1_down_box[2]]
            # cv2.imwrite(location_save_dir / f"camera1_{camera1_timestamp}--up_target.jpg", camera1_up_target)
            # cv2.imwrite(location_save_dir / f"camera1_{camera1_timestamp}--down_target.jpg", camera1_down_target)

            try:
                logger.info(f"camera 1 up rings location start")
                camera1_up_result = adaptive_threshold_rings_location(
                    camera1_up_target,
                    f"camera1--{camera1_timestamp}--up",
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
                camera1_up_result['metadata'] = camera1_metadata
                logger.success(f"{camera1_up_result = }")
                logger.success(f"camera 1 up rings location success")
                camera1_up_center: list[float] = [camera1_up_result['center_x_mean'], camera1_up_result['center_y_mean']]

                string = json.dumps(camera1_up_result, ensure_ascii=False)
                with open(camera1_result_save_path, mode='a', encoding='utf-8') as f:
                    f.write(string + "\n")
            except Exception as e:
                camera1_up_center = None
                logger.error(e)
                logger.error(f"camera 1 up circle location failed")

            try:
                logger.info(f"camera 1 down rings location start")
                camera1_down_result = adaptive_threshold_rings_location(
                    camera1_down_target,
                    f"camera1--{camera1_timestamp}--down",
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
                camera1_down_result['metadata'] = camera1_metadata
                logger.success(f"{camera1_down_result = }")
                logger.success(f"camera 1 down rings location success")
                camera1_down_center: list[float] = [camera1_down_result['center_x_mean'], camera1_down_result['center_y_mean']]

                string = json.dumps(camera1_down_result, ensure_ascii=False)
                with open(camera1_result_save_path, mode='a', encoding='utf-8') as f:
                    f.write(string + "\n")
            except Exception as e:
                camera1_down_center = None
                logger.error(e)
                logger.error(f"camera 1 down circle location failed")
        #-------------------- camera1 --------------------#
        #------------------------- 检测目标 -------------------------#

        #------------------------- 后处理 -------------------------#
        if all([camera0_up_center, camera0_down_center, camera1_up_center, camera1_down_center]):
            logger.critical(f"all cameras location success")
            logger.info(f"{camera0_up_center = }, {camera0_down_center = }, {camera1_up_center = }, {camera1_down_center = }")

            # 畸变矫正

            # 检查距离

            # 传递消息

            # 清零检测结果
            camera0_up_center = None
            camera0_down_center = None
            camera1_up_center = None
            camera1_down_center = None
        #------------------------- 后处理 -------------------------#

        # 主循环休眠
        main_sleep_interval: int = MainConfig.getattr("main_sleep_interval")
        time.sleep(main_sleep_interval / 1000)

        # 测试调整相机
        if i > 10000:
            break
        logger.warning(f"{i = }")
        i += 1


if __name__ == "__main__":
    main()
