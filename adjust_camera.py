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
    suitable_ignore_ratio: float = 0.0,
) -> tuple[float, int]:
    mean_bright = mean_brightness(image)
    logger.info(f"{exposure_time = }, {mean_bright = }")

    # 缩小 mean_light_suitable_range 区间，让它更加宽松
    suitable_range = mean_light_suitable_range[1] - mean_light_suitable_range[0]
    ignore_range = suitable_range * suitable_ignore_ratio
    low = mean_light_suitable_range[0] + ignore_range
    high = mean_light_suitable_range[1] - ignore_range

    if mean_bright < low:
        return exposure_time + adjust_exposure_time_step, 1
    elif mean_bright > high:
        return exposure_time - adjust_exposure_time_step, -1
    else:
        return exposure_time, 0


# 先使用高分辨率，后使用低分辨率
def adjust_exposure1(
    camera_queue: queue.Queue,
    boxes: list[list[int]] | None = None, # [[x1, y1, x2, y2]]
) -> dict[int, list[list[int]] | None]:
    logger.info("adjust exposure start")
    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
    exposure2boxes: dict[int, list[list[int]] | None] = {
        CameraConfig.getattr("exposure_time"): boxes
    }

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

        # 全图
        if boxes is None:
            # 先检测曝光是否合适
            new_exposure_time, direction = adjust_exposure_by_mean(
                image,
                image_metadata['ExposureTime'],
                AdjustCameraConfig.getattr("mean_light_suitable_range"),
                AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                AdjustCameraConfig.getattr("suitable_ignore_ratio"),
            )
            logger.info(f"{new_exposure_time = }, {direction = }")
            if direction == 0:
                logger.success(f"full picture original exposure time {new_exposure_time} us is ok")
                exposure2boxes = {
                    new_exposure_time: boxes
                }
                return

        # 分组
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
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                directions.append(direction)
                new_exposure_times.append(new_exposure_time)
                logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

            if all(direction == 0 for direction in directions):
                exposure2boxes = {
                    int(np.mean(new_exposure_times)): boxes
                }
                logger.success(f"boxes: {boxes}, original exposure time {new_exposure_times[0]} us is ok")
                return

    except queue.Empty:
        logger.error("get picture timeout")

    #-------------------- 第一次全分辨率拍摄 --------------------#
    logger.warning(f"original exposure time is not ok, new_exposure_time = {new_exposure_time} us")
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

            # 全图
            if _boxes is None:
                new_exposure_time, direction = adjust_exposure_by_mean(
                    image,
                    image_metadata['ExposureTime'],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                CameraConfig.setattr("exposure_time", new_exposure_time)
                logger.info(f"{new_exposure_time = }, {direction = }")

                if direction == 0:
                    exposure2boxes = {
                        new_exposure_time: boxes
                    }
                    break

            # 分组
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
                        AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                    )

                    directions.append(direction)
                    new_exposure_times.append(new_exposure_time)
                    logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

                logger.info(f"{new_exposure_times = }")
                new_exposure_mean = int(np.mean(new_exposure_times))
                logger.info(f"{new_exposure_mean = }")
                CameraConfig.setattr("exposure_time", new_exposure_mean)

                if all(direction == 0 for direction in directions):
                    exposure2boxes = {
                        new_exposure_mean: boxes
                    }
                    break

        except queue.Empty:
            logger.error("get picture timeout")

        if j == adjust_total_times - 1:
            logger.warning(f"adjust exposure times: {adjust_total_times}, final failed")

    # 还原相机配置
    CameraConfig.setattr("capture_mode", default_capture_mode)
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)
    #-------------------- 低分辨率快速拍摄 --------------------#

    logger.success(f"final set {new_exposure_time = }")
    logger.info(f"{exposure2boxes = }")
    logger.info("adjust exposure end")
    return exposure2boxes


# 始终使用高分辨率
def adjust_exposure2(
    camera_queue: queue.Queue,
    boxes: list[list[int]] | None = None, # [[x1, y1, x2, y2]]
) -> dict[int, list[list[int]] | None]:
    logger.info("adjust exposure start")
    boxes: np.ndarray | None = np.array(boxes) if boxes is not None else None
    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
    exposure2boxes: dict[int, list[list[int]] | None] = {
        CameraConfig.getattr("exposure_time"): boxes
    }

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

            # 全图
            if boxes is None:
                new_exposure_time, direction = adjust_exposure_by_mean(
                    image,
                    image_metadata['ExposureTime'],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                CameraConfig.setattr("exposure_time", new_exposure_time)
                logger.info(f"{new_exposure_time = }, {direction = }")

                if direction == 0:
                    exposure2boxes = {
                        new_exposure_time: boxes
                    }
                    if j == 0:
                        logger.success(f"full picture original exposure time {new_exposure_time} us is ok")
                    break

            # 划分box
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
                        AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                    )

                    directions.append(direction)
                    new_exposure_times.append(new_exposure_time)
                    logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

                logger.info(f"{new_exposure_times = }")
                new_exposure_mean = int(np.mean(new_exposure_times))
                logger.info(f"{new_exposure_mean = }")
                CameraConfig.setattr("exposure_time", new_exposure_mean)

                if all(direction == 0 for direction in directions):
                    exposure2boxes = {
                        new_exposure_times[0]: boxes
                    }
                    if j == 0:
                        logger.success(f"boxes: {boxes}, original exposure time {new_exposure_times} us is ok")
                    break

        except queue.Empty:
            logger.error("get picture timeout")

        if j == adjust_total_times - 1:
            logger.warning(f"adjust exposure times: {adjust_total_times}, final failed")

    # 还原相机配置
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)

    #-------------------- 高分辨率快速拍摄 --------------------#

    logger.success(f"final set {new_exposure_time = }")
    logger.info(f"{exposure2boxes = }")
    logger.info("adjust exposure end")
    return exposure2boxes


# 使用递归实现快速调节曝光
def adjust_exposure3(
    camera_queue: queue.Queue,
    boxes: list[list[int]] | None = None, # [[x1, y1, x2, y2]]
) -> dict[int, list[list[int]] | None]:
    logger.info("adjust exposure start")
    boxes: np.ndarray | None = np.array(boxes) if boxes is not None else None
    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
    exposure2boxes: dict[int, list[list[int]] | None] = {
        CameraConfig.getattr("exposure_time"): boxes
    }

    #-------------------- 高分辨率快速拍摄 --------------------#
    default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
    default_return_image_time_interval: int = CameraConfig.getattr("return_image_time_interval")

    # 调整相机配置，加快拍照
    CameraConfig.setattr("capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval"))
    CameraConfig.setattr("return_image_time_interval", AdjustCameraConfig.getattr("return_image_time_interval"))

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

        # 全图
        if boxes is None:
            new_exposure_time, direction = adjust_exposure_by_mean(
                image,
                image_metadata['ExposureTime'],
                AdjustCameraConfig.getattr("mean_light_suitable_range"),
                AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                AdjustCameraConfig.getattr("suitable_ignore_ratio"),
            )
            CameraConfig.setattr("exposure_time", new_exposure_time)

            # 可以
            if direction == 0:
                exposure2boxes = {
                    new_exposure_time: boxes
                }
                logger.success(f"full picture exposure time {new_exposure_time} us is ok")
            # 需要调节
            else:
                logger.info(f"{new_exposure_time = }, {direction = }")
                exposure2boxes = adjust_exposure3(camera_queue)

        # 划分box
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
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                directions.append(direction)
                new_exposure_times.append(new_exposure_time)
                logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

            logger.info(f"{new_exposure_times = }")

            # 全都可以
            if all(direction == 0 for direction in directions):
                exposure2boxes = {
                    new_exposure_times[0]: boxes
                }
                logger.success(f"boxes: {boxes}, original exposure time {new_exposure_times[0]} us is ok")
            # 需要分组
            else:
                # 将box分组
                directions = np.array(directions)
                directions_low = np.where(directions==-1)[0]
                directions_ok = np.where(directions==0)[0]
                directions_high = np.where(directions==1)[0]

                # 分组使用递归调用
                exposure2boxes = {}
                if directions_low.size > 0:
                    logger.info("adjust exposure low")
                    exposure_time_low = new_exposure_times[directions_low[0]]
                    boxes_low = boxes[directions_low]
                    CameraConfig.setattr("exposure_time", exposure_time_low)
                    exposure2boxes_low = adjust_exposure3(camera_queue, boxes_low)
                    exposure2boxes.update(exposure2boxes_low)

                if directions_ok.size > 0:
                    exposure_time_ok = new_exposure_times[directions_ok[0]]
                    boxes_ok = boxes[directions_ok]
                    exposure2boxes_ok = {
                        exposure_time_ok: boxes_ok
                    }
                    exposure2boxes.update(exposure2boxes_ok)

                if directions_high.size > 0:
                    logger.info("adjust exposure high")
                    boxes_high = boxes[directions_high]
                    exposure_time_high = new_exposure_times[directions_high[0]]
                    CameraConfig.setattr("exposure_time", exposure_time_high)
                    exposure2boxes_high = adjust_exposure3(camera_queue, boxes_high)
                    exposure2boxes.update(exposure2boxes_high)

    except queue.Empty:
        logger.error("get picture timeout")

    # 还原相机配置
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)
    #-------------------- 高分辨率快速拍摄 --------------------#

    logger.success(f"{exposure2boxes = }")
    logger.info("adjust exposure end")
    return exposure2boxes
