import time
from datetime import datetime
from queue import Queue
from loguru import logger
import cv2

from algorithm import RaspberryCameras
from config import MainConfig, CameraConfig
from utils import enhance_contrast_clahe


# 相机线程
def camera_engine(
    queue: Queue,
    camera_index: int = 0,
    *args,
    **kwargs,
):
    # 启动相机函数
    def start_camera_engine():
        log_file_path = (
            MainConfig.getattr("log_dir")
            / f"camera_{camera_index}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )
        raspberry_cameras = RaspberryCameras(
            camera_index,
            MainConfig.getattr("log_level"),
            log_file_path,
            CameraConfig.getattr("low_res_ratio"),
        )
        raspberry_cameras.start(camera_index)
        return raspberry_cameras

    raspberry_cameras = start_camera_engine()
    # 存放当前的 mode
    capture_mode: str = CameraConfig.getattr("capture_mode")
    raspberry_cameras.switch_mode(camera_index, capture_mode)

    return_before_time = time.time()
    while True:
        try:
            captime_time_begin = time.time()

            # 1.判断是否需要切换模式
            _capture_mode: str = CameraConfig.getattr("capture_mode")
            if capture_mode != _capture_mode:
                capture_mode = _capture_mode
                raspberry_cameras.switch_mode(camera_index, capture_mode)

            # 2.拍照
            # 曝光时间, 增益
            if camera_index == 0:
                exposure_time: int | None = CameraConfig.getattr("camera0_exposure_time")
                analogue_gain: float | None = CameraConfig.getattr("camera0_analogue_gain")
            else:
                exposure_time: int | None = CameraConfig.getattr("camera1_exposure_time")
                analogue_gain: float | None = CameraConfig.getattr("camera0_analogue_gain")
            # 曝光时间宽容度
            exposure_time_tolerance_percent: float = CameraConfig.getattr(
                "exposure_time_tolerance_percent"
            )
            if exposure_time is not None:
                exposure_time_tolerance = (
                    exposure_time * exposure_time_tolerance_percent
                )
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
            image, metadata = raspberry_cameras.capture(
                camera_index,
                exposure_time,
                analogue_gain,
                timestamp,
            )
            # logger.info(f"camera {camera_index} capture {timestamp} image")

            # 3.判断是否返回图片
            return_image_time_interval = CameraConfig.getattr(
                "return_image_time_interval"
            )
            return_current_time = time.time()
            # 取整为时间周期
            _return_before_time_period = int(
                return_before_time * 1000 // return_image_time_interval
            )
            _return_current_time_period = int(
                return_current_time * 1000 // return_image_time_interval
            )
            # 曝光在设定的时间范围内就返回
            if _return_current_time_period > _return_before_time_period:
                _ExposureTime = metadata["ExposureTime"]
                # 默认 exposure_time 为 None,要手动设定
                if exposure_time is None:
                    exposure_time = _ExposureTime
                    exposure_time_tolerance = (
                        exposure_time * exposure_time_tolerance_percent
                    )
                    if camera_index == 0:
                        CameraConfig.setattr("camera0_exposure_time", exposure_time)
                    else:
                        CameraConfig.setattr("camera1_exposure_time", exposure_time)
                if (
                    _ExposureTime >= exposure_time - exposure_time_tolerance
                    and _ExposureTime <= exposure_time + exposure_time_tolerance
                ):
                    # 如果队列满了，就删除队列的第一张图片
                    if queue.full():
                        queue.get()
                        logger.warning(
                            f"camera {camera_index} queue is full, delete the first image in queue"
                        )

                    # 转换为Gray，Default: RGB
                    output_format = CameraConfig.getattr("output_format")
                    if output_format == "gray":
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                    # 是否有滤光板
                    has_filter_plate = CameraConfig.getattr("has_filter_plate")
                    if not has_filter_plate:
                        image = enhance_contrast_clahe(image)

                    # 将图像放入队列中
                    queue.put((timestamp, image, metadata))
                    logger.success(
                        f"camera {camera_index} put {timestamp} image into queue"
                    )
                    return_before_time = return_current_time

            # sleep
            captime_time_end = time.time()

            capture_time_interval = CameraConfig.getattr("capture_time_interval")
            sleep_time = max(
                0,
                capture_time_interval / 1000 - (captime_time_end - captime_time_begin),
            )
            logger.info(f"camera {camera_index} {sleep_time = } s")
            time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"camera {camera_index} error: {e}")

        else:
            # 相机超时重启
            get_picture_timeout_threshold: int = MainConfig.getattr(
                "get_picture_timeout_threshold"
            )
            get_picture_timeout_count: int = MainConfig.getattr(
                "get_picture_timeout_count"
            )
            if get_picture_timeout_count >= get_picture_timeout_threshold:
                logger.warning(
                    f"get picture timeout count: {get_picture_timeout_count} >= threshold: {get_picture_timeout_threshold}, restart camera"
                )
                # 关闭相机
                raspberry_cameras.close(camera_index)
                raspberry_cameras.close_log_file()
                # 重启相机
                raspberry_cameras = start_camera_engine()
                raspberry_cameras.switch_mode(camera_index, capture_mode)
