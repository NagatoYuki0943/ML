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
    raspberry_cameras = RaspberryCameras(camera_index)
    raspberry_cameras.start(camera_index)

    # 存放当前的 mode
    capture_mode: str = CameraConfig.getattr("capture_mode")
    raspberry_cameras.switch_mode(camera_index, capture_mode)

    return_time1 = time.time()
    while True:
        captime_time1 = time.time()

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
        logger.info(f"camera {camera_index} capture {timestamp} image")

        # 3.判断是否返回图片
        return_image_time_interval = CameraConfig.getattr("return_image_time_interval")
        reutrn_time2 = time.time()
        if reutrn_time2 - return_time1 > return_image_time_interval / 1000:
            # 曝光在设定的时间范围内就返回
            _ExposureTime = metadata['ExposureTime']
            if _ExposureTime >= exposure_time - 100 and _ExposureTime <= exposure_time + 100:
                # 如果队列满了，就删除队列的第一张图片
                if queue.full():
                    queue.get()
                    logger.warning(f"camera {camera_index} queue is full, delete the first image in queue")
                # 将图像放入队列中
                queue.put((timestamp, image, metadata))
                logger.success(f"camera {camera_index} put {timestamp} image into queue")
                return_time1 = time.time()

        # sleep
        captime_time2 = time.time()

        capture_time_interval = CameraConfig.getattr("capture_time_interval")
        sleep_time = max(0, capture_time_interval / 1000 - (captime_time2 - captime_time1))
        logger.info(f"camera {camera_index} {sleep_time = } s")
        time.sleep(sleep_time)
