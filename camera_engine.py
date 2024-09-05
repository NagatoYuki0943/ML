import time
from datetime import datetime
from queue import Queue
from loguru import logger

from algorithm import RaspberryCameras
from config import CameraConfig


# 相机线程
def camera_engine(
    queue: Queue,
    camera_index: int = 0,
    *args,
    **kwargs,
):
    raspberry_cameras = RaspberryCameras(
        camera_index,
        low_res_ratio=CameraConfig.getattr("low_res_ratio"),
    )
    raspberry_cameras.start(camera_index)

    # 存放当前的 mode
    capture_mode: str = CameraConfig.getattr("capture_mode")
    raspberry_cameras.switch_mode(camera_index, capture_mode)

    return_before_time = time.time()
    while True:
        captime_time_begin = time.time()

        # 1.判断是否需要切换模式
        _capture_mode: str = CameraConfig.getattr("capture_mode")
        if capture_mode != _capture_mode:
            capture_mode = _capture_mode
            raspberry_cameras.switch_mode(camera_index, capture_mode)

        # 2.拍照
        exposure_time: int = CameraConfig.getattr("exposure_time")
        analogue_gain: float = CameraConfig.getattr("analogue_gain")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        image, metadata = raspberry_cameras.capture(
            camera_index,
            exposure_time,
            analogue_gain,
            timestamp,
        )
        # logger.info(f"camera {camera_index} capture {timestamp} image")

        # 3.判断是否返回图片
        return_image_time_interval = CameraConfig.getattr("return_image_time_interval")
        return_current_time = time.time()
        # 取整为时间周期
        _return_before_time_period = int(return_before_time * 1000 // return_image_time_interval)
        _return_current_time_period = int(return_current_time * 1000 // return_image_time_interval)
        # 曝光在设定的时间范围内就返回
        if _return_current_time_period > _return_before_time_period:
            _ExposureTime = metadata['ExposureTime']
            # 默认 exposure_time 为 None,要手动设定
            if exposure_time is None:
                exposure_time = _ExposureTime
                CameraConfig.setattr("exposure_time", exposure_time)
            if _ExposureTime >= exposure_time - 100 and _ExposureTime <= exposure_time + 100:
                # 如果队列满了，就删除队列的第一张图片
                if queue.full():
                    queue.get()
                    logger.warning(f"camera {camera_index} queue is full, delete the first image in queue")
                # 将图像放入队列中
                queue.put((timestamp, image, metadata))
                logger.success(f"camera {camera_index} put {timestamp} image into queue")
                return_before_time = return_current_time

        # sleep
        captime_time_end = time.time()

        capture_time_interval = CameraConfig.getattr("capture_time_interval")
        sleep_time = max(0, capture_time_interval / 1000 - (captime_time_end - captime_time_begin))
        logger.info(f"camera {camera_index} {sleep_time = } s")
        time.sleep(sleep_time)
