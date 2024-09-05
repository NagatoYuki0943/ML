import numpy as np
import cv2
from algorithm import mean_brightness
from config import MainConfig, CameraConfig, AdjustCameraConfig
import queue
from loguru import logger


def adjust_exposure_by_mean(
    image: np.ndarray,
    exposure_time: float,
    mean_light_suitable_range: tuple[float],
    adjust_exposure_time_step: float = 1000,
) -> tuple[float, int]:
    mean_bright = mean_brightness(image)
    if mean_bright < mean_light_suitable_range[0]:
        return exposure_time + adjust_exposure_time_step, 1
    elif mean_bright > mean_light_suitable_range[1]:
        return exposure_time - adjust_exposure_time_step, -1
    else:
        return exposure_time, 0


# 先使用高分辨率，后使用低分辨率
def adjust_exposure1(
    camera_queue: queue.Queue,
    boxes: list[list[int]] | None = None, # [[x1, y1, x2, y2]]
) -> list[int]:
    logger.info("adjust exposure start")
    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
    final_exposure_time: int = CameraConfig.getattr("exposure_time")

    #-------------------- 第一次全分辨率拍摄 --------------------#
    # 第一次拍摄不调整相机的 capture_mode，因此使用的是原始分辨率
    try:
        camera_qsize = camera_queue.qsize()
        if camera_qsize > 1:
            logger.warning(f"camera got {camera_qsize} frames, ignore {camera_qsize - 1} frames")
            for _ in range(camera_qsize - 1):
                try:
                    camera_queue.get(timeout=get_picture_timeout)
                except queue.Empty:
                    logger.error("get picture timeout")

        _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if boxes is None:
            # 不截图
            # 先检测曝光是否合适
            new_exposure_time, direction = adjust_exposure_by_mean(
                image,
                image_metadata['ExposureTime'],
                AdjustCameraConfig.getattr("mean_light_suitable_range"),
                AdjustCameraConfig.getattr("adjust_exposure_time_step"),
            )
            logger.info(f"{new_exposure_time = }, {direction = }")
            if direction == 0:
                logger.success("original exposure is ok")
                final_exposure_time = new_exposure_time
                return

        else:
            directions = []
            new_exposure_times = []
            # 循环boxes
            for i, box in enumerate(boxes):
                target_image = image[box[1]:box[3], box[0]:box[2]]
                    # 先检测曝光是否合适
                new_exposure_time, direction = adjust_exposure_by_mean(
                    target_image,
                    image_metadata['ExposureTime'],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                )

                directions.append(direction)
                new_exposure_times.append(new_exposure_time)
                logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

            if all(direction == 0 for direction in directions):
                final_exposure_time = int(np.mean(new_exposure_times))
                logger.success("original exposure is ok")
                return

    except queue.Empty:
        logger.error("get picture timeout")

    #-------------------- 第一次全分辨率拍摄 --------------------#
    logger.warning(f"original exposure is not ok, new_exposure_time = {new_exposure_time}")
    CameraConfig.setattr("exposure_time", new_exposure_time)

    #-------------------- 低分辨率快速拍摄 --------------------#
    default_capture_mode: str = CameraConfig.getattr("capture_mode")
    default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
    default_return_image_time_interval: int = CameraConfig.getattr("return_image_time_interval")
    low_res_ratio: float = CameraConfig.getattr("low_res_ratio")

    _boxes = [[int(i * low_res_ratio) for i in box] for box in boxes] if boxes is not None else None

    # 调整相机配置，加快拍照
    CameraConfig.setattr("capture_mode", AdjustCameraConfig.getattr("capture_mode"))
    CameraConfig.setattr("capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval"))
    CameraConfig.setattr("return_image_time_interval", AdjustCameraConfig.getattr("return_image_time_interval"))

    # 忽略所有的相机拍照
    while not camera_queue.empty():
        camera_queue.get(timeout=get_picture_timeout)

    # 调整次数上限
    adjust_total_times = AdjustCameraConfig.getattr("adjust_total_times")
    for j in range(adjust_total_times):
        try:
            camera_qsize = camera_queue.qsize()
            if camera_qsize > 1:
                logger.warning(f"camera got {camera_qsize} frames, ignore {camera_qsize - 1} frames")
                for _ in range(camera_qsize - 1):
                    try:
                        camera_queue.get(timeout=get_picture_timeout)
                    except queue.Empty:
                        logger.error("get picture timeout")

            _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if _boxes is None:
                new_exposure_time, direction = adjust_exposure_by_mean(
                    image,
                    image_metadata['ExposureTime'],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                )

                CameraConfig.setattr("exposure_time", new_exposure_time)
                logger.info(f"{new_exposure_time = }, {direction = }")

                if direction == 0:
                    final_exposure_time = new_exposure_time
                    break

            else:
                directions = []
                new_exposure_times = []
                for i, _box in enumerate(_boxes):
                    target_image = image[_box[1]:_box[3], _box[0]:_box[2]]
                    new_exposure_time, direction = adjust_exposure_by_mean(
                        target_image,
                        image_metadata['ExposureTime'],
                        AdjustCameraConfig.getattr("mean_light_suitable_range"),
                        AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                    )

                    directions.append(direction)
                    new_exposure_times.append(new_exposure_time)
                    logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

                logger.info(f"{new_exposure_times = }")
                new_exposure_mean = int(np.mean(new_exposure_times))
                logger.info(f"{new_exposure_mean = }")
                CameraConfig.setattr("exposure_time", new_exposure_mean)

                if all(direction == 0 for direction in directions):
                    final_exposure_time = new_exposure_mean
                    break

        except queue.Empty:
            logger.error("get picture timeout")

        if j == adjust_total_times - 1:
            logger.warning(f"adjust exposure times: {adjust_total_times}, final failed")

    logger.success(f"final set {new_exposure_time = }")

    # 还原相机配置
    CameraConfig.setattr("capture_mode", default_capture_mode)
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)
    #-------------------- 低分辨率快速拍摄 --------------------#

    logger.info("adjust exposure end")
    return [final_exposure_time] if isinstance(final_exposure_time, int) else final_exposure_time


# 始终使用高分辨率
def adjust_exposure2(
    camera_queue: queue.Queue,
    boxes: list[list[int]] | None = None, # [[x1, y1, x2, y2]]
) -> list[int]:
    logger.info("adjust exposure start")
    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
    final_exposure_time: int = CameraConfig.getattr("exposure_time")

    #-------------------- 高分辨率快速拍摄 --------------------#
    default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
    default_return_image_time_interval: int = CameraConfig.getattr("return_image_time_interval")

    # 调整相机配置，加快拍照
    CameraConfig.setattr("capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval"))
    CameraConfig.setattr("return_image_time_interval", AdjustCameraConfig.getattr("return_image_time_interval"))

    # 调整次数上限
    adjust_total_times = AdjustCameraConfig.getattr("adjust_total_times")
    for j in range(adjust_total_times):
        try:
            camera_qsize = camera_queue.qsize()
            if camera_qsize > 1:
                logger.warning(f"camera got {camera_qsize} frames, ignore {camera_qsize - 1} frames")
                for _ in range(camera_qsize - 1):
                    try:
                        camera_queue.get(timeout=get_picture_timeout)
                    except queue.Empty:
                        logger.error("get picture timeout")

            image_timestamp, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
            logger.info(f"camera get image: {image_timestamp}, ExposureTime = {image_metadata['ExposureTime']}, AnalogueGain = {image_metadata['AnalogueGain']}, shape = {image.shape}")
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if boxes is None:
                new_exposure_time, direction = adjust_exposure_by_mean(
                    image,
                    image_metadata['ExposureTime'],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                )

                CameraConfig.setattr("exposure_time", new_exposure_time)
                logger.info(f"{new_exposure_time = }, {direction = }")

                if direction == 0:
                    final_exposure_time = new_exposure_time
                    if j == 0:
                        logger.success("original exposure is ok")
                    break

            else:
                directions = []
                new_exposure_times = []
                for i, box in enumerate(boxes):
                    target_image = image[box[1]:box[3], box[0]:box[2]]
                    new_exposure_time, direction = adjust_exposure_by_mean(
                        target_image,
                        image_metadata['ExposureTime'],
                        AdjustCameraConfig.getattr("mean_light_suitable_range"),
                        AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                    )

                    directions.append(direction)
                    new_exposure_times.append(new_exposure_time)
                    logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

                logger.info(f"{new_exposure_times = }")
                new_exposure_mean = int(np.mean(new_exposure_times))
                logger.info(f"{new_exposure_mean = }")
                CameraConfig.setattr("exposure_time", new_exposure_mean)

                if all(direction == 0 for direction in directions):
                    final_exposure_time = new_exposure_mean
                    if j == 0:
                        logger.success("original exposure is ok")
                    break

        except queue.Empty:
            logger.error("get picture timeout")

        if j == adjust_total_times - 1:
            logger.warning(f"adjust exposure times: {adjust_total_times}, final failed")

    logger.success(f"final set {new_exposure_time = }")

    # 还原相机配置
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)
    #-------------------- 高分辨率快速拍摄 --------------------#

    logger.info("adjust exposure end")
    return [final_exposure_time] if isinstance(final_exposure_time, int) else final_exposure_time
