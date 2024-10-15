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
    StereoCalibration,
    RaspberryMQTT,
    RaspberrySerialPort,
    sort_boxes_center,
    pixel_num2object_size,
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


# 将日志输出到文件
# 每天 0 点新创建一个 log 文件
handler_id = logger.add(
    str(MainConfig.getattr("loguru_log_path")),
    level=MainConfig.getattr("log_level"),
    rotation="00:00",
)


# ------------------------------ 初始化 ------------------------------ #
logger.info("init start")

# -------------------- 基础 -------------------- #
# 主线程消息队列
main_queue = queue.Queue()
image0_timestamp: str
image0: np.ndarray
image0_metadata: dict
# -------------------- 基础 -------------------- #

# -------------------- 运行时配置 -------------------- #

# -------------------- 运行时配置 -------------------- #

# -------------------- 初始化相机 -------------------- #
logger.info("开始初始化相机")
CAMERA_INDEX = 0
camera0_thread = ThreadWrapper(
    target_func=camera_engine,
    queue_maxsize=CameraConfig.getattr("queue_maxsize"),
    camera_index=CAMERA_INDEX,
)
camera0_thread.start()
camera0_queue = camera0_thread.queue

time.sleep(1)
logger.success("初始化相机完成")
# -------------------- 初始化相机 -------------------- #

# -------------------- 畸变矫正 -------------------- #
logger.info("开始初始化畸变矫正")
stereo_calibration = StereoCalibration(
    StereoCalibrationConfig.getattr("camera_matrix_left"),
    StereoCalibrationConfig.getattr("camera_matrix_right"),
    StereoCalibrationConfig.getattr("distortion_coefficients_left"),
    StereoCalibrationConfig.getattr("distortion_coefficients_right"),
    StereoCalibrationConfig.getattr("R"),
    StereoCalibrationConfig.getattr("T"),
    StereoCalibrationConfig.getattr("pixel_width_mm"),
)
logger.success("初始化畸变矫正完成")
# -------------------- 畸变矫正 -------------------- #

# -------------------- 初始化串口 -------------------- #
# logger.info("开始初始化串口")
# serial_objects = []

# for port in [SerialCommConfig.getattr("camera0_ser_port"), SerialCommConfig.getattr("camera1_ser_port")]:
#     if port:
#         object = RaspberrySerialPort(
#             SerialCommConfig.getattr("temperature_data_save_path"),
#             port,
#             SerialCommConfig.getattr("baudrate"),
#             SerialCommConfig.getattr("timeout"),
#             SerialCommConfig.getattr("BUFFER_SIZE"),
#         )
#         serial_objects.append(object)

# serial_send_thread = ThreadWrapper(
#     target_func=serial_send,
#     serial_ports=serial_objects,
# )
# serial_receive_thread = Thread(
#     target=serial_receive,
#     kwargs={
#         "serial_ports": serial_objects,
#         "queue": main_queue,
#     },
# )
# serial_send_queue = serial_send_thread.queue
# serial_receive_thread.start()
# serial_send_thread.start()
# logger.success("初始化串口完成")
# -------------------- 初始化串口 -------------------- #

# -------------------- 初始化MQTT客户端 -------------------- #
# logger.info("开始初始化MQTT客户端")
# mqtt_comm = RaspberryMQTT(
#     MQTTConfig.getattr("broker"),
#     MQTTConfig.getattr("port"),
#     MQTTConfig.getattr("timeout"),
#     MQTTConfig.getattr("topic"),
#     MQTTConfig.getattr("username"),
#     MQTTConfig.getattr("password"),
#     MQTTConfig.getattr("clientId"),
#     MQTTConfig.getattr("apikey"),
# )
# mqtt_send_thread = ThreadWrapper(
#     target_func=mqtt_send,
#     queue_maxsize=MQTTConfig.getattr("send_queue_maxsize"),
#     client=mqtt_comm,
# )
# mqtt_send_queue = mqtt_send_thread.queue
# mqtt_receive_thread = Thread(
#     target=mqtt_receive,
#     kwargs={
#         "client": mqtt_comm,
#         "main_queue": main_queue,
#         "send_queue": mqtt_send_queue,
#     },
# )
# mqtt_receive_thread.start()
# mqtt_send_thread.start()
# logger.success("初始化MQTT客户端完成")
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
# mqtt_send_queue.put(send_msg)


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
    adjust_exposure_full_res(camera0_queue, camera0_id2boxstate, True, CAMERA_INDEX)
    logger.success("camera0 ajust exposure end")
    try:
        _, image0, image0_metadata = camera0_queue.get(timeout=get_picture_timeout)
        image_path = save_dir / "image0_adjust_exposure.jpg"
        save_image(image0, image_path)
        logger.info(f"save `image0 adjust exposure` image to {image_path}")
    except queue.Empty:
        get_picture_timeout_process()
    # -------------------- camera0 -------------------- #
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
                rectified_image0, CAMERA_INDEX
            )
        else:
            logger.success("camera0_id2boxstate is not None, use find_around_target")
            camera0_id2boxstate, camera0_got_target_number = find_around_target(
                rectified_image0, CAMERA_INDEX
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
    adjust_led_level_param = (
        {
            "level": 1,  # need_darker - (达到最低就代表关闭补光灯), need_lighter +
            "times": 10,  # 亮的时间
        },
    )
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

                # ------------------------- 调整全图曝光 ------------------------- #
                # --------------- camera0 --------------- #
                logger.info("full image0 ajust exposure start")

                # 每次使用补光灯调整曝光的总次数
                adjust_with_flash_total_times: int = AdjustCameraConfig.getattr(
                    "adjust_with_flash_total_times"
                )
                adjust_with_falsh_total_time = 0
                while True:
                    # 如果上一次使用了补光灯，那这一次也使用补光灯
                    if use_flash:
                        Send.send_adjust_led_level_msg(adjust_led_level_param)

                    # 调整曝光
                    camera0_id2boxstate: dict[int, dict] | None = (
                        MatchTemplateConfig.getattr("camera0_id2boxstate")
                    )
                    _, need_darker, need_lighter = adjust_exposure_full_res(
                        camera0_queue,
                        camera0_id2boxstate,
                        True,
                        CAMERA_INDEX,
                    )

                    # ---------- 补光灯 ---------- #
                    if need_darker or need_lighter:
                        use_flash = True
                        if need_darker:
                            # 已经是最低的补光灯
                            if adjust_led_level_param["level"] <= 1:
                                # 关闭补光灯
                                use_flash = False
                                logger.warning(
                                    "already is the lowest flash, close flash"
                                )
                                # TODO: 关闭补光灯
                                raise NotImplementedError(
                                    "no need flash, close flash not implemented"
                                )
                            else:
                                # 降低补光灯亮度
                                adjust_led_level_param["level"] -= 1
                        else:
                            # 已经是最高的补光灯
                            if adjust_led_level_param["level"] >= 10:
                                logger.warning(
                                    "already is the highest flash, can't adjust flash"
                                )
                                continue
                            else:
                                # 增加补光灯亮度
                                adjust_led_level_param["level"] += 1
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
                        rectified_image0, CAMERA_INDEX
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
                        CAMERA_INDEX,
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

                    # -------------------- 设定循环参数 -------------------- #
                    # 所有相机循环曝光时间
                    cycle_exposure_times = camera0_cycle_exposure_times
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
                        # TODO: 处理其他相机
                        pass
                    # -------------------- box location -------------------- #
                else:
                    # 没有需要曝光的图片, cycle_loop_count 也要加1, 因为默认的总循环次数为2
                    cycle_loop_count += 1

                # 正常判断是否结束周期
                if cycle_loop_count >= total_cycle_loop_count - 1:
                    # ------------------------- 整理检测结果 ------------------------- #
                    logger.info("last cycle, try to compare and save results")
                    logger.success(f"{camera0_cycle_results = }")
                    # 保存到文件
                    save_to_jsonl({"camera0": camera0_cycle_results}, history_save_path)

                    # 防止值不存在
                    send_msg_data = {}

                    target_number: int = MatchTemplateConfig.getattr("target_number")

                    camera0_standard_results: dict | None = RingsLocationConfig.getattr(
                        "camera0_standard_results"
                    )
                    # 初始化标准结果
                    if (
                        camera0_standard_results is None
                        or len(camera0_standard_results) != target_number
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

                        # -------------------- send result -------------------- #
                        if camera0_standard_results is not None:
                            RingsLocationConfig.setattr(
                                "camera0_standard_results", camera0_standard_results
                            )
                            # 发送初始坐标数据结果
                            # ✅️✅️✅️ 正常数据消息 ✅️✅️✅️
                            logger.success("send init data message.")
                            send_msg_data = {
                                "L3_WK_1": 20,  # 温度传感器
                                "L3_WK_2": 20,
                                "L3_WK_3": 20,
                                "L3_WK_4": 20,
                                "L3_WK_5": 20,
                                "L3_WK_6": 20,
                                "L3_WK_7": 20,
                                "L3_WK_8": 20,
                            }
                            _send_msg_data = {
                                f"L1_SJ_{k+1}": {"X": 0, "Y": 0}
                                for k in camera0_standard_results.keys()
                            }
                            send_msg_data.update(_send_msg_data)
                            logger.info(f"send_msg_data: {send_msg_data}")
                            send_msg = {
                                "cmd": "update",
                                "did": MQTTConfig.getattr("did"),
                                "data": send_msg_data,
                            }
                            # mqtt_send_queue.put(send_msg)
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

                        # -------------------- send msg -------------------- #

                        send_msg_data = {
                            "L3_WK_1": 20,  # 温度传感器
                            "L3_WK_2": 20,
                            "L3_WK_3": 20,
                            "L3_WK_4": 20,
                            "L3_WK_5": 20,
                            "L3_WK_6": 20,
                            "L3_WK_7": 20,
                            "L3_WK_8": 20,
                        }
                        _send_msg_data = {
                            f"L1_SJ_{k+1}": {"X": v[0], "Y": v[1]}
                            for k, v in camera0_distance_result.items()
                        }
                        send_msg_data.update(_send_msg_data)
                        logger.info(f"send_msg_data: {send_msg_data}")
                        if len(camera0_over_distance_ids) > 0:
                            # ⚠️⚠️⚠️ 有box移动距离超过阈值 ⚠️⚠️⚠️
                            logger.warning(
                                f"box {camera0_over_distance_ids} move distance is over threshold."
                            )

                            # 保存位移的图片
                            image_path = save_dir / "target_displacement.jpg"
                            save_image(image0, image_path)
                            logger.info(
                                f"save `target displacement` image to {image_path}"
                            )
                            # 位移告警消息
                            send_msg = {
                                "cmd": "alarm",
                                "body": {
                                    "did": MQTTConfig.getattr("did"),
                                    "type": "displacement",
                                    "at": get_now_time(),
                                    "number": [
                                        i + 1 for i in camera0_over_distance_ids
                                    ],  # 表示异常的靶标编号
                                    "data": send_msg_data,
                                    "path": [str(image_path)],  # 图片本地路径
                                    "img": ["target_displacement.jpg"],  # 文件名称
                                },
                            }
                            # mqtt_send_queue.put(send_msg)
                        else:
                            # ✅️✅️✅️ 所有 box 移动距离都小于阈值 ✅️✅️✅️
                            logger.success("All box move distance is under threshold.")
                            # 正常数据消息
                            send_msg = {
                                "cmd": "update",
                                "did": MQTTConfig.getattr("did"),
                                "data": send_msg_data,
                            }
                            # mqtt_send_queue.put(send_msg)
                        # -------------------- send msg -------------------- #

                    # ------------------------- 整理检测结果 ------------------------- #

                    # ------------------------- 检查是否丢失目标 ------------------------- #
                    target_number = MatchTemplateConfig.getattr("target_number")

                    # -------------------- camera0 lost box -------------------- #
                    camera0_got_target_number = MatchTemplateConfig.getattr(
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
                            find_lost_target(rectified_image0, CAMERA_INDEX)
                            target_number = MatchTemplateConfig.getattr("target_number")
                            camera0_got_target_number: int = MatchTemplateConfig.getattr(
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
                                image_path = save_dir / "target_loss.jpg"
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
                                        "img": ["target_loss.jpg"],  # 文件名称
                                    },
                                }
                                # mqtt_send_queue.put(send_msg)
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
                    # ------------------------- 检查是否丢失目标 ------------------------- #

                    # ------------------------- 结束周期 ------------------------- #
                    # 重置周期内循环计数
                    cycle_loop_count = -1

                    if use_flash:
                        # 关闭补光灯
                        # TODO: 关闭补光灯
                        raise NotImplementedError("close flash")

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
        logger.warning(f"{i = }")
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

        # 靶标校正消息
        elif cmd == "targetcorrection":
            Receive.receive_target_correction_msg(received_msg)

        elif cmd == "deletetarget":
            Receive.receive_delete_target_msg(received_msg)

        elif cmd == "settarget":
            Receive.receive_set_target_msg(received_msg)

        # 参考靶标设定消息
        elif cmd == "setreferencetarget":
            Receive.receive_set_reference_target_msg(received_msg)

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
        global camera0_cycle_results

        logger.info("device deploying, reset config and init target")
        # 设备部署，重置配置和初始靶标
        load_config_from_yaml(config_path=original_config_path)
        # 重设检测的结果(用于发送新的消息)
        camera0_cycle_results = {}
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
            # mqtt_send_queue.put(send_msg)
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
            # mqtt_send_queue.put(send_msg)
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
            logger.info(
                f"`upload image` get image success, image_timestamp: {image0_timestamp}"
            )
            # 保存图片
            image_path = save_dir / "upload_image.jpg"
            save_image(image0, image_path)
            logger.info(f"save `upload image` success, save image to {image_path}")

            # 现场图像查询响应消息
            send_msg = {
                "cmd": "getimage",
                "body": {
                    "code": 200,
                    "did": MQTTConfig.getattr("did"),
                    "at": get_now_time(),
                    "msg": "upload succeed",
                    "path": [str(image_path)],  # 图片本地路径
                    "img": ["upload_image.jpg"],  # 文件名称
                },
                "msgid": "bb6f3eeb2",
            }
            # mqtt_send_queue.put(send_msg)
            logger.success(f"upload image send msg success, image_path: {image_path}")
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
            # mqtt_send_queue.put(send_msg)
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
        logger.success("received askadjusttempdata response")
        received_temp_control_msg = True

    @staticmethod
    def receive_temp_data_msg(received_msg: dict | None = None):
        """日常温度数据"""
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
        logger.success("received temp data")

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
        logger.success("received adjust temp data")

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
        logger.success("received stop adjust temp data")
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
        # mqtt_send_queue.put(send_msg)
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
            new_config_path = Path(received_msg["body"]["path"])
            # 载入配置文件
            load_config_from_yaml(config_path=new_config_path)

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
            # mqtt_send_queue.put(send_msg)
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
            # mqtt_send_queue.put(send_msg)
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
        # mqtt_send_queue.put(send_msg)
        logger.success(
            f"get config file success, config_path: {save_dir / 'config_runtime.yaml'}"
        )


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
        data = {
            f"L1_SJ_{k+1}": {
                "box": v["box"],
                "X": v["center"][0],
                "Y": v["center"][1],
                "Z": v["distance"],
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

        if not camera0_cycle_results:
            logger.warning(
                "camera0_cycle_results is empty, can't send device deploying msg, wait for next cycle."
            )
            return

        try:
            # 获取照片
            _, image0, _ = camera0_queue.get(timeout=get_picture_timeout)
            image_path = save_dir / "deploy.jpg"
            save_image(image0, image_path)
            logger.info(f"save `deploying image` success, save image to {image_path}")
            path = [str(image_path)]
            img = ["deploy.jpg"]
        except queue.Empty:
            path = []
            img = []
            get_picture_timeout_process()

        _data = Send.get_xyz(camera0_cycle_results)
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
        # mqtt_send_queue.put(send_msg)
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
        # mqtt_send_queue.put(send_msg)
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
        # mqtt_send_queue.put(send_msg)
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
        # mqtt_send_queue.put(send_msg)
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
        sensor_state = {
            "L3_WK_1": 0,  # 0表示无错误，-1供电异常，
            "L3_WK_2": 0,  # -2传感器数据异常，-3采样间隔内没有采集到数据
            "L3_WK_3": 0,
            "L3_WK_4": 0,
            "L3_WK_5": 0,
            "L3_WK_6": -1,
        }

        camera0_standard_results: dict | None = RingsLocationConfig.getattr(
            "camera0_standard_results"
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

        sensor_state.update(box_sensor_state)

        # 设备状态查询响应消息
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
        # mqtt_send_queue.put(send_msg)
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
        # mqtt_send_queue.put(send_msg)
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
        # mqtt_send_queue.put(send_msg)
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
        # mqtt_send_queue.put(send_msg)
        logger.warning("send device error msg")

    @staticmethod
    def send_temperature_change_msg():
        """设备温控变化消息"""
        global is_temp_stable
        send_msg = {
            "cmd": "devicestate",
            "body": {
                "did": MQTTConfig.getattr("did"),
                "type": "temperature_control",
                "at": get_now_time(),
                "data": {
                    "L3_WK_1": 20,  # 内仓实际温度
                    "control_t": 30,  # 目标温度
                    "control_way": "warm/cold",
                    "pwm_data": 10,
                },
            },
        }
        # mqtt_send_queue.put(send_msg)
        logger.success("send temperature change msg")
        is_temp_stable = False

    @staticmethod
    def send_temperature_control_msg():
        global received_temp_control_msg
        global is_temp_stable

        """温度控制命令"""
        logger.info("send temperature control msg")
        send_msg = {"cmd": "adjusttempdata", "param": {"control_t": 10}, "msgid": 1}
        # serial_send_queue.put(send_msg)
        logger.success("send temperature control msg success")
        received_temp_control_msg = False
        is_temp_stable = False

    @staticmethod
    def send_adjust_led_level_msg(adjust_led_level_param: dict):
        logger.info("send adjust led level msg")
        # 补光灯控制命令
        send_msg = {
            "cmd": "adjustLEDlevel",
            "param": adjust_led_level_param,
            "msgid": 1,
        }
        # serial_send_queue.put(send_msg)
        logger.success("send adjust led level msg success")

        while True:
            received_msg: dict = main_queue.get(timeout=10)
            cmd: str = received_msg.get("cmd")
            # 温控板回复补光灯调节指令
            if cmd == "askadjustLEDlevel":
                # {
                #     "cmd": "askadjustLEDlevel",
                #     "times": "2024-09-11T15:45:30",
                #     "param":{
                #         "result": "OK/NOT"
                #     },
                #     "msgid": 1
                # }
                logger.success("received askadjustLEDlevel response")
                break
            else:
                Receive.switch(received_msg)


if __name__ == "__main__":
    main()
