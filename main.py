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
    StereoCalibration,
    RaspberryMQTT,
    RaspberrySerialPort,
    RaspberryFTP,
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
    FTPConfig,
)

from camera_engine import camera_engine
from find_target import find_target
from adjust_camera import adjust_exposure
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
    main_queue = Queue()

    save_dir: Path = MainConfig.getattr("save_dir")
    location_save_dir = MainConfig.getattr("location_save_dir")
    left_camera_result_save_path = MainConfig.getattr("left_camera_result_save_path")
    right_camera_result_save_path = MainConfig.getattr("right_camera_result_save_path")
    calibration_result_save_path = MainConfig.getattr("calibration_result_save_path")
    get_picture_timeout = MainConfig.getattr("get_picture_timeout")
    #-------------------- 基础 --------------------#

    #-------------------- 初始化相机 --------------------#
    logger.info("开始初始化相机")
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

    # 确定左右相机 index
    if CameraConfig.camera_left_index == 0:
        left_camera_thread = camera0_thread
        right_camera_thread = camera1_thread
    else:
        left_camera_thread = camera1_thread
        right_camera_thread = camera0_thread
    left_camera_queue = left_camera_thread.queue
    right_camera_queue = right_camera_thread.queue

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
    logger.info("开始初始化串口")
    serial_objects = []

    for port in SerialCommConfig.getattr('ports'):
        object = RaspberrySerialPort(
            SerialCommConfig.getattr('temperature_data_save_path'),
            port,
            SerialCommConfig.getattr('baudrate'),
            SerialCommConfig.getattr('timeout'),
            SerialCommConfig.getattr('BUFFER_SIZE'),
            SerialCommConfig.getattr('LOG_SIZE'),
        )
        serial_objects.append(object)

    serial_send_thread = ThreadWrapper(
        target_func = serial_send,
        ser = serial_objects,
    )
    serial_receive_thread = Thread(
        target = serial_receive,
        kwargs={
            'ser':serial_objects,
            'queue':main_queue,
        },
    )
    serial_send_queue = serial_send_thread.queue
    serial_receive_thread.start()
    serial_send_thread.start()
    logger.success("初始化串口完成")
    #-------------------- 初始化串口 --------------------#

    #-------------------- 初始化MQTT客户端 --------------------#
    logger.info("开始初始化MQTT客户端")
    mqtt_comm = RaspberryMQTT(
        MQTTConfig.getattr('broker'),
        MQTTConfig.getattr('port'),
        MQTTConfig.getattr('timeout'),
        MQTTConfig.getattr('topic'),
        MQTTConfig.getattr('username'),
        MQTTConfig.getattr('password'),
        MQTTConfig.getattr('clientId'),
        MainConfig.getattr('apikey'),
    )
    # FTP客户端
    ftp = RaspberryFTP(
        FTPConfig.getattr('ip'),
        FTPConfig.getattr('port'),
        FTPConfig.getattr('username'),
        FTPConfig.getattr('password'),
    )
    mqtt_send_thread = ThreadWrapper(
        target_func = mqtt_send,
        client = mqtt_comm,
        ftp = ftp
    )
    mqtt_send_queue = mqtt_send_thread.queue
    mqtt_receive_thread = Thread(
        target = mqtt_receive,
        kwargs={
            'client':mqtt_comm,
            'ftp':ftp,
            'main_queue':main_queue,
            'send_queue':mqtt_send_queue,
        },
    )
    mqtt_receive_thread.start()
    mqtt_send_thread.start()
    logger.success("初始化MQTT客户端完成")
    #-------------------- 初始化MQTT客户端 --------------------#

    logger.success("init end")
    #------------------------------ 初始化 ------------------------------#

    #------------------------------ 调整曝光1 ------------------------------#
    adjust_exposure(left_camera_queue)
    #------------------------------ 调整曝光1 ------------------------------#

    #------------------------------ 找到目标 ------------------------------#
    logger.info("find target start")
    left_image_timestamp: str
    right_image_timestamp: str
    left_image: np.ndarray
    right_image: np.ndarray
    left_image_metadata: dict
    right_image_metadata: dict

    #-------------------- 取图 --------------------#
    _, left_image, left_image_metadata = left_camera_queue.get()
    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
    logger.warning(f"{left_image.shape = }, ExposureTime = {left_image_metadata['ExposureTime']}, AnalogueGain = {left_image_metadata['AnalogueGain']}")
    cv2.imwrite(save_dir / "left_image_standard.jpg", left_image)

    _, right_image, right_image_metadata = right_camera_queue.get()
    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
    logger.warning(f"{right_image.shape = }, ExposureTime = {right_image_metadata['ExposureTime']}, AnalogueGain = {right_image_metadata['AnalogueGain']}")
    cv2.imwrite(save_dir / "right_image_standard.jpg", right_image)
    #-------------------- 取图 --------------------#

    #-------------------- 畸变矫正 --------------------#
    logger.info("rectify image start")
    rectified_left_image, rectified_right_image, \
    R1, R2, P1, P2, Q, roi1, roi2, \
    undistorted_left_image, undistorted_right_image = stereo_calibration.rectify_images(
        left_image,
        right_image,
    )
    cv2.imwrite(save_dir / "left_image_undistorted.jpg", undistorted_left_image)
    cv2.imwrite(save_dir / "right_image_undistorted.jpg", undistorted_right_image)
    logger.success("rectify image success")
    #-------------------- 畸变矫正 --------------------#

    #-------------------- left image --------------------#
    logger.info("left image find target start")
    left_boxes = find_target(undistorted_left_image)
    MatchTemplateConfig.left_boxes = left_boxes
    logger.info(f"left image find target result: \n{left_boxes}")

    # 绘制boxes
    left_image_draw = undistorted_left_image.copy()
    for i in range(len(left_boxes)):
        cv2.rectangle(
            img = left_image_draw,
            pt1 = (left_boxes[i][0], left_boxes[i][1]),
            pt2 = (left_boxes[i][2], left_boxes[i][3]),
            color = (255, 0, 0),
            thickness = 3
        )
    plt.figure(figsize=(10, 10))
    plt.imshow(left_image_draw, cmap='gray')
    plt.savefig(save_dir / "left_image_match_template.png")
    plt.close()
    logger.success("left image find target success")
    #-------------------- left image --------------------#

    #-------------------- right image --------------------#
    logger.info("right image find target start")
    right_boxes = find_target(undistorted_right_image)
    MatchTemplateConfig.right_boxes = right_boxes
    logger.info(f"right image find target result: \n{right_boxes}")

    # 绘制boxes
    right_image_draw = undistorted_right_image.copy()
    for i in range(len(right_boxes)):
        cv2.rectangle(
            img = right_image_draw,
            pt1 = (right_boxes[i][0], right_boxes[i][1]),
            pt2 = (right_boxes[i][2], right_boxes[i][3]),
            color = (255, 0, 0),
            thickness = 3
        )
    plt.figure(figsize=(10, 10))
    plt.imshow(right_image_draw, cmap='gray')
    plt.savefig(save_dir / "right_image_match_template.png")
    plt.close()
    logger.success("right image find target success")
    #-------------------- right image --------------------#
    logger.success("find target end")
    #------------------------------ 找到目标 ------------------------------#

    #------------------------------ 调整曝光2 ------------------------------#
    adjust_exposure(left_camera_queue, left_boxes[0])
    # 调整曝光后，清空队列，清除调整曝光前的图片
    clear_queue(left_camera_queue, right_camera_queue)
    _, left_image, _ = left_camera_queue.get()
    cv2.imwrite(save_dir / "left_image_adjust_exposure.jpg", left_image)
    _, right_image, _ = right_camera_queue.get()
    cv2.imwrite(save_dir / "right_image_adjust_exposure.jpg", right_image)
    #------------------------------ 调整曝光2 ------------------------------#

    # 主循环
    i = 0

    while True:
        # 每次开始调整曝光
        adjust_exposure(left_camera_queue, left_boxes[0])
        # logger.warning(f"{left_image.__class__.__name__}, {right_image.__class__.__name__}")
        #-------------------- left camera capture --------------------#
        left_camera_qsize = left_camera_queue.qsize()
        # 忽略多余的图片
        if left_camera_qsize > 1:
            logger.warning(f"left camera got {left_camera_qsize} frames, ignore {left_camera_qsize - 1} frames")
            for _ in range(left_camera_qsize - 1):
                left_camera_queue.get()

        # 获取照片
        left_image_timestamp, left_image, left_image_metadata = left_camera_queue.get()
        logger.info(f"left camera get image: {left_image_timestamp}, shape = {left_image.shape}, ExposureTime = {left_image_metadata['ExposureTime']}, AnalogueGain = {left_image_metadata['AnalogueGain']}")
        #-------------------- left camera capture --------------------#

        #-------------------- right camera capture --------------------#
        right_camera_qsize = right_camera_queue.qsize()
        # 忽略多余的图片
        if right_camera_qsize > 1:
            logger.warning(f"right camera got {right_camera_qsize} frames, ignore {right_camera_qsize - 1} frames")
            for _ in range(right_camera_qsize - 1):
                right_camera_queue.get()

        # 获取照片
        right_image_timestamp, right_image, right_image_metadata = right_camera_queue.get()
        logger.info(f"right camera get image: {right_image_timestamp}, shape = {right_image.shape}, ExposureTime = {right_image_metadata['ExposureTime']}, AnalogueGain = {right_image_metadata['AnalogueGain']}")
        #-------------------- right camera capture --------------------#

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
        rectified_left_image, rectified_right_image, \
        R1, R2, P1, P2, Q, roi1, roi2, \
        undistorted_left_image, undistorted_right_image = stereo_calibration.rectify_images(
            left_image,
            right_image,
        )
        #-------------------- 畸变矫正 --------------------#

        left_image_up_center = None
        left_image_down_center = None
        right_image_up_center = None
        right_image_down_center = None
        #-------------------- left image location --------------------#
        # 截取图像
        left_boxes = MatchTemplateConfig.left_boxes
        left_image_up_target = undistorted_left_image[left_boxes[0][1]:left_boxes[0][3], left_boxes[0][0]:left_boxes[0][2]]
        left_image_down_target = undistorted_left_image[left_boxes[1][1]:left_boxes[1][3], left_boxes[1][0]:left_boxes[1][2]]
        # cv2.imwrite(location_save_dir / f"left_image_{left_image_timestamp}--up_target.jpg", left_image_up_target)
        # cv2.imwrite(location_save_dir / f"left_image_{left_image_timestamp}--down_target.jpg", left_image_down_target)

        try:
            logger.info(f"left image up rings location start")
            left_image_up_result = adaptive_threshold_rings_location(
                left_image_up_target,
                f"left_image--{left_image_timestamp}--up",
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
            left_image_up_result['metadata'] = left_image_metadata
            logger.success(f"{left_image_up_result = }")
            logger.success(f"left image up rings location success")
            left_image_up_center = np.array([left_image_up_result['center_x_mean'], left_image_up_result['center_y_mean']])
            # 保存到文件
            string = json.dumps(left_image_up_result, ensure_ascii=False)
            with open(left_camera_result_save_path, mode='a', encoding='utf-8') as f:
                f.write(string + "\n")
        except Exception as e:
            left_image_up_center = None
            logger.error(e)
            logger.error(f"left image up circle location failed")

        try:
            logger.info(f"left image down rings location start")
            left_image_down_result = adaptive_threshold_rings_location(
                left_image_down_target,
                f"left_image--{left_image_timestamp}--down",
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
            left_image_down_result['metadata'] = left_image_metadata
            logger.success(f"{left_image_down_result = }")
            logger.success(f"left image down rings location success")
            left_image_down_center = np.array([left_image_down_result['center_x_mean'], left_image_down_result['center_y_mean']])
            # 保存到文件
            string = json.dumps(left_image_down_result, ensure_ascii=False)
            with open(left_camera_result_save_path, mode='a', encoding='utf-8') as f:
                f.write(string + "\n")
        except Exception as e:
            left_image_down_center = None
            logger.error(e)
            logger.error(f"left image down circle location failed")
        #-------------------- left image location --------------------#

        #-------------------- right image location --------------------#
        # 截取图像
        right_boxes = MatchTemplateConfig.right_boxes
        right_image_up_target = undistorted_right_image[right_boxes[0][1]:right_boxes[0][3], right_boxes[0][0]:right_boxes[0][2]]
        right_image_down_target = undistorted_right_image[right_boxes[1][1]:right_boxes[1][3], right_boxes[1][0]:right_boxes[1][2]]
        # cv2.imwrite(location_save_dir / f"right_image_{right_image_timestamp}--up_target.jpg", right_image_up_target)
        # cv2.imwrite(location_save_dir / f"right_image_{right_image_timestamp}--down_target.jpg", right_image_down_target)

        try:
            logger.info(f"right image up rings location start")
            right_image_up_result = adaptive_threshold_rings_location(
                right_image_up_target,
                f"right_image--{right_image_timestamp}--up",
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
            right_image_up_result['metadata'] = right_image_metadata
            logger.success(f"{right_image_up_result = }")
            logger.success(f"right image up rings location success")
            right_image_up_center = np.array([right_image_up_result['center_x_mean'], right_image_up_result['center_y_mean']])
            # 保存到文件
            string = json.dumps(right_image_up_result, ensure_ascii=False)
            with open(right_camera_result_save_path, mode='a', encoding='utf-8') as f:
                f.write(string + "\n")
        except Exception as e:
            right_image_up_center = None
            logger.error(e)
            logger.error(f"right image up circle location failed")

        try:
            logger.info(f"right image down rings location start")
            right_image_down_result = adaptive_threshold_rings_location(
                right_image_down_target,
                f"right_image--{right_image_timestamp}--down",
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
            right_image_down_result['metadata'] = right_image_metadata
            logger.success(f"{right_image_down_result = }")
            logger.success(f"right image down rings location success")
            right_image_down_center = np.array([right_image_down_result['center_x_mean'], right_image_down_result['center_y_mean']])
            # 保存到文件
            string = json.dumps(right_image_down_result, ensure_ascii=False)
            with open(right_camera_result_save_path, mode='a', encoding='utf-8') as f:
                f.write(string + "\n")
        except Exception as e:
            right_image_down_center = None
            logger.error(e)
            logger.error(f"right image down circle location failed")
        #-------------------- right image location --------------------#

        if all(center is not None for center in \
            [left_image_up_center, left_image_down_center, right_image_up_center, right_image_down_center]):
            logger.critical(f"all cameras location success")
            logger.info(f"{left_image_up_center = }, {left_image_down_center = }, {right_image_up_center = }, {right_image_down_center = }")

            # 矫正坐标
            left_points = np.array([left_image_up_center, left_image_down_center])
            rectified_left_points = stereo_calibration.undistort_points(
                left_points,
                stereo_calibration.camera_matrix_left,
                R1, P1
            )
            right_points = np.array([right_image_up_center, right_image_down_center])
            rectified_right_points = stereo_calibration.undistort_points(
                right_points,
                stereo_calibration.camera_matrix_right,
                R2, P2
            )
            logger.info(f"{rectified_left_points = }, {rectified_right_points = }")

            # 保存到文件
            string = json.dumps({
                    "left_image_timestamp": left_image_timestamp,
                    "right_image_timestamp": right_image_timestamp,
                    "rectified_left_points": rectified_left_points,
                    "rectified_right_points": rectified_right_points,
                },
                ensure_ascii=False
            )
            with open(calibration_result_save_path, mode='a', encoding='utf-8') as f:
                f.write(string + "\n")

            # 检查距离

            # 传递消息

        #------------------------- 检测目标 -------------------------#

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
