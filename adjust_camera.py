import numpy as np
import cv2
from algorithm import mean_brightness
from config import MainConfig, CameraConfig, AdjustCameraConfig
import queue
from loguru import logger

from utils import clear_queue


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


# # 先使用高分辨率，后使用低分辨率
# def adjust_exposure1(
#     camera_queue: queue.Queue,
#     boxes: list[list[int]] | None = None, # [[x1, y1, x2, y2]]
# ) -> dict[int, list[list[int]] | None]:
#     logger.info("adjust exposure start")
#     get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
#     exposure2boxes: dict[int, list[list[int]] | None] = {
#         CameraConfig.getattr("exposure_time"): boxes
#     }

#     #-------------------- 第一次全分辨率拍摄 --------------------#
#     # 第一次拍摄不调整相机的 capture_mode，因此使用的是原始分辨率
#     try:
#         camera_qsize = camera_queue.qsize()
#         if camera_qsize > 1:
#             logger.warning(f"camera got {camera_qsize} frames, ignore {camera_qsize - 1} frames")
#             for _ in range(camera_qsize - 1):
#                 try:
#                     camera_queue.get(timeout=get_picture_timeout)
#                 except queue.Empty:
#                     logger.error("get picture timeout")

#         _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)

#         # 全图
#         if boxes is None:
#             # 先检测曝光是否合适
#             new_exposure_time, direction = adjust_exposure_by_mean(
#                 image,
#                 image_metadata['ExposureTime'],
#                 AdjustCameraConfig.getattr("mean_light_suitable_range"),
#                 AdjustCameraConfig.getattr("adjust_exposure_time_step"),
#                 AdjustCameraConfig.getattr("suitable_ignore_ratio"),
#             )
#             logger.info(f"{new_exposure_time = }, {direction = }")
#             if direction == 0:
#                 logger.success(f"full picture original exposure time {new_exposure_time} us is ok")
#                 exposure2boxes = {
#                     new_exposure_time: boxes
#                 }
#                 return

#         # 分组
#         else:
#             directions = []
#             new_exposure_times = []
#             # 循环boxes
#             for i, box in enumerate(boxes):
#                 target_image = image[box[1]:box[3], box[0]:box[2]]
#                     # 先检测曝光是否合适
#                 new_exposure_time, direction = adjust_exposure_by_mean(
#                     target_image,
#                     image_metadata['ExposureTime'],
#                     AdjustCameraConfig.getattr("mean_light_suitable_range"),
#                     AdjustCameraConfig.getattr("adjust_exposure_time_step"),
#                     AdjustCameraConfig.getattr("suitable_ignore_ratio"),
#                 )

#                 directions.append(direction)
#                 new_exposure_times.append(new_exposure_time)
#                 logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

#             if all(direction == 0 for direction in directions):
#                 exposure2boxes = {
#                     int(np.mean(new_exposure_times)): boxes
#                 }
#                 logger.success(f"boxes: {boxes}, original exposure time {new_exposure_times[0]} us is ok")
#                 return

#     except queue.Empty:
#         logger.error("get picture timeout")

#     #-------------------- 第一次全分辨率拍摄 --------------------#
#     logger.warning(f"original exposure time is not ok, new_exposure_time = {new_exposure_time} us")
#     CameraConfig.setattr("exposure_time", new_exposure_time)

#     #-------------------- 低分辨率快速拍摄 --------------------#
#     default_capture_mode: str = CameraConfig.getattr("capture_mode")
#     default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
#     default_return_image_time_interval: int = CameraConfig.getattr("return_image_time_interval")
#     low_res_ratio: float = CameraConfig.getattr("low_res_ratio")

#     _boxes = [[int(i * low_res_ratio) for i in box] for box in boxes] if boxes is not None else None

#     # 调整相机配置，加快拍照
#     CameraConfig.setattr("capture_mode", AdjustCameraConfig.getattr("capture_mode"))
#     CameraConfig.setattr("capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval"))
#     CameraConfig.setattr("return_image_time_interval", AdjustCameraConfig.getattr("return_image_time_interval"))

#     # 忽略所有的相机拍照
#     while not camera_queue.empty():
#         camera_queue.get(timeout=get_picture_timeout)

#     # 调整次数上限
#     adjust_total_times = AdjustCameraConfig.getattr("adjust_total_times")
#     for j in range(adjust_total_times):
#         try:
#             camera_qsize = camera_queue.qsize()
#             if camera_qsize > 1:
#                 logger.warning(f"camera got {camera_qsize} frames, ignore {camera_qsize - 1} frames")
#                 for _ in range(camera_qsize - 1):
#                     try:
#                         camera_queue.get(timeout=get_picture_timeout)
#                     except queue.Empty:
#                         logger.error("get picture timeout")

#             _, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#             # 全图
#             if _boxes is None:
#                 new_exposure_time, direction = adjust_exposure_by_mean(
#                     image,
#                     image_metadata['ExposureTime'],
#                     AdjustCameraConfig.getattr("mean_light_suitable_range"),
#                     AdjustCameraConfig.getattr("adjust_exposure_time_step"),
#                     AdjustCameraConfig.getattr("suitable_ignore_ratio"),
#                 )

#                 CameraConfig.setattr("exposure_time", new_exposure_time)
#                 logger.info(f"{new_exposure_time = }, {direction = }")

#                 if direction == 0:
#                     exposure2boxes = {
#                         new_exposure_time: boxes
#                     }
#                     break

#             # 分组
#             else:
#                 directions = []
#                 new_exposure_times = []
#                 for i, _box in enumerate(_boxes):
#                     target_image = image[_box[1]:_box[3], _box[0]:_box[2]]
#                     new_exposure_time, direction = adjust_exposure_by_mean(
#                         target_image,
#                         image_metadata['ExposureTime'],
#                         AdjustCameraConfig.getattr("mean_light_suitable_range"),
#                         AdjustCameraConfig.getattr("adjust_exposure_time_step"),
#                         AdjustCameraConfig.getattr("suitable_ignore_ratio"),
#                     )

#                     directions.append(direction)
#                     new_exposure_times.append(new_exposure_time)
#                     logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

#                 logger.info(f"{new_exposure_times = }")
#                 new_exposure_mean = int(np.mean(new_exposure_times))
#                 logger.info(f"{new_exposure_mean = }")
#                 CameraConfig.setattr("exposure_time", new_exposure_mean)

#                 if all(direction == 0 for direction in directions):
#                     exposure2boxes = {
#                         new_exposure_mean: boxes
#                     }
#                     break

#         except queue.Empty:
#             logger.error("get picture timeout")

#         if j == adjust_total_times - 1:
#             logger.warning(f"adjust exposure times: {adjust_total_times}, final failed")

#     # 还原相机配置
#     CameraConfig.setattr("capture_mode", default_capture_mode)
#     CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
#     CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)
#     #-------------------- 低分辨率快速拍摄 --------------------#

#     logger.success(f"final set {new_exposure_time = }")
#     logger.info(f"{exposure2boxes = }")
#     logger.info("adjust exposure end")
#     return exposure2boxes


# # 始终使用高分辨率
# def adjust_exposure2(
#     camera_queue: queue.Queue,
#     boxes: list[list[int]] | None = None, # [[x1, y1, x2, y2]]
# ) -> dict[int, list[list[int]] | None]:
#     logger.info("adjust exposure start")
#     boxes: np.ndarray | None = np.array(boxes) if boxes is not None else None
#     get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
#     exposure2boxes: dict[int, list[list[int]] | None] = {
#         CameraConfig.getattr("exposure_time"): boxes
#     }

#     #-------------------- 高分辨率快速拍摄 --------------------#
#     default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
#     default_return_image_time_interval: int = CameraConfig.getattr("return_image_time_interval")

#     # 调整相机配置，加快拍照
#     CameraConfig.setattr("capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval"))
#     CameraConfig.setattr("return_image_time_interval", AdjustCameraConfig.getattr("return_image_time_interval"))

#     # 调整次数上限
#     adjust_total_times = AdjustCameraConfig.getattr("adjust_total_times")
#     for j in range(adjust_total_times):
#         try:
#             camera_qsize = camera_queue.qsize()
#             if camera_qsize > 1:
#                 logger.warning(f"camera got {camera_qsize} frames, ignore {camera_qsize - 1} frames")
#                 for _ in range(camera_qsize - 1):
#                     try:
#                         camera_queue.get(timeout=get_picture_timeout)
#                     except queue.Empty:
#                         logger.error("get picture timeout")

#             image_timestamp, image, image_metadata = camera_queue.get(timeout=get_picture_timeout)
#             logger.info(f"camera get image: {image_timestamp}, ExposureTime = {image_metadata['ExposureTime']}, AnalogueGain = {image_metadata['AnalogueGain']}, shape = {image.shape}")

#             # 全图
#             if boxes is None:
#                 new_exposure_time, direction = adjust_exposure_by_mean(
#                     image,
#                     image_metadata['ExposureTime'],
#                     AdjustCameraConfig.getattr("mean_light_suitable_range"),
#                     AdjustCameraConfig.getattr("adjust_exposure_time_step"),
#                     AdjustCameraConfig.getattr("suitable_ignore_ratio"),
#                 )

#                 CameraConfig.setattr("exposure_time", new_exposure_time)
#                 logger.info(f"{new_exposure_time = }, {direction = }")

#                 if direction == 0:
#                     exposure2boxes = {
#                         new_exposure_time: boxes
#                     }
#                     if j == 0:
#                         logger.success(f"full picture original exposure time {new_exposure_time} us is ok")
#                     break

#             # 划分box
#             else:
#                 directions = []
#                 new_exposure_times = []
#                 for i, box in enumerate(boxes):
#                     target_image = image[box[1]:box[3], box[0]:box[2]]
#                     new_exposure_time, direction = adjust_exposure_by_mean(
#                         target_image,
#                         image_metadata['ExposureTime'],
#                         AdjustCameraConfig.getattr("mean_light_suitable_range"),
#                         AdjustCameraConfig.getattr("adjust_exposure_time_step"),
#                         AdjustCameraConfig.getattr("suitable_ignore_ratio"),
#                     )

#                     directions.append(direction)
#                     new_exposure_times.append(new_exposure_time)
#                     logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

#                 logger.info(f"{new_exposure_times = }")
#                 new_exposure_mean = int(np.mean(new_exposure_times))
#                 logger.info(f"{new_exposure_mean = }")
#                 CameraConfig.setattr("exposure_time", new_exposure_mean)

#                 if all(direction == 0 for direction in directions):
#                     exposure2boxes = {
#                         new_exposure_times[0]: boxes
#                     }
#                     if j == 0:
#                         logger.success(f"boxes: {boxes}, original exposure time {new_exposure_times} us is ok")
#                     break

#         except queue.Empty:
#             logger.error("get picture timeout")

#         if j == adjust_total_times - 1:
#             logger.warning(f"adjust exposure times: {adjust_total_times}, final failed")

#     # 还原相机配置
#     CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
#     CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)

#     #-------------------- 高分辨率快速拍摄 --------------------#

#     logger.success(f"final set {new_exposure_time = }")
#     logger.info(f"{exposure2boxes = }")
#     logger.info("adjust exposure end")
#     return exposure2boxes


def adjust_exposure_full_res_recursive(
    camera_queue: queue.Queue,
    id2boxstate: dict | None = None,
) -> dict[int, dict | None]:
    """使用递归实现快速调节曝光，全程使用高分辨率拍摄

    Args:
        camera_queue (queue.Queue): 相机队列
        id2boxstate (dict | None, optional): id对应box的状态，包括box和状态. Defaults to None.
            {
                i: {
                    "ratio": ratio,
                    "score": score,
                    "box": [x1, y1, x2, y2]
                },
                ...
            }

    Returns:
        dict[int, dict | None]: 曝光对应不同的box状态
            {
                exposure_time: {
                    i: {
                        "ratio": ratio,
                        "score": score,
                        "box": [x1, y1, x2, y2]
                    },
                    ...
                },
                ...
            }
    """
    logger.info("adjust exposure start")

    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
    exposure2id2boxstate: dict[int, dict | None] = {
        CameraConfig.getattr("exposure_time"): id2boxstate
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
        if id2boxstate is None:
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
                exposure2id2boxstate = {
                    new_exposure_time: id2boxstate
                }
                logger.success(f"full picture exposure time {new_exposure_time} us is ok")
            # 需要调节
            else:
                logger.info(f"{new_exposure_time = }, {direction = }")
                exposure2id2boxstate = adjust_exposure_full_res_recursive(camera_queue)

        # 划分box
        else:
            # directions 和 new_exposure_times 的 key 对应 boxid
            directions = {}
            new_exposure_times = {}
            for i, boxstate in id2boxstate.items():
                box = boxstate["box"]
                # 空box不处理
                if box is None:
                    continue
                target_image = image[box[1]:box[3], box[0]:box[2]]
                new_exposure_time, direction = adjust_exposure_by_mean(
                    target_image,
                    image_metadata['ExposureTime'],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                directions[i] = direction
                new_exposure_times[i] = new_exposure_time
                logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

            logger.info(f"{new_exposure_times = }")

            # 所有 box 都合适
            if all(direction == 0 for direction in directions.values()):
                new_exposure_time = list(new_exposure_times.values())[0]
                exposure2id2boxstate = {
                    new_exposure_time: id2boxstate
                }
                logger.success(f"id2boxstate: {id2boxstate}, original exposure time {new_exposure_time} us is ok")
            # box 需要分组
            else:
                # exampe:
                #   directions 和 new_exposure_times 的 key 对应 boxid
                #   directions = { 1: 0, 3: 1, 4: -1, 5: 0, 7: 1}
                #   new_exposure_times = { 1: 10, 3: 11, 4: 9, 5: 10, 7: 11}
                #   directions_keys = [1, 3, 4, 5, 7]
                #   directions_values = [0, 1, -1, 0, 1]
                # 根据 directions_values 分为3组, 获取对应的index, 然后根据index获取keys, keys对应boxid

                # 将box分组
                # 根据 directions 判断曝光
                directions_keys = np.array(list(directions.keys()))
                directions_values = np.array(list(directions.values()))
                # 获取 values 对应的 index
                directions_low_index = np.where(directions_values==-1)[0]
                directions_ok_index = np.where(directions_values==0)[0]
                directions_high_index = np.where(directions_values==1)[0]
                # 根据 index 获取 keys, keys 对应 boxid
                directions_low_keys = directions_keys[directions_low_index]
                directions_ok_keys = directions_keys[directions_ok_index]
                directions_high_keys = directions_keys[directions_high_index]

                # 分组使用递归调用
                exposure2id2boxstate = {}
                if directions_low_keys.size > 0:
                    logger.info("adjust exposure low")
                    # 获取对应的曝光值
                    exposure_time_low = new_exposure_times[directions_low_keys[0]]
                    CameraConfig.setattr("exposure_time", exposure_time_low)
                    # 根据 keys 获取 boxstate
                    id2boxstate_low = {int(i): id2boxstate[i] for i in directions_low_keys}
                    exposure2boxes_low = adjust_exposure_full_res_recursive(camera_queue, id2boxstate_low)
                    exposure2id2boxstate.update(exposure2boxes_low)

                if directions_ok_keys.size > 0:
                    # 获取对应的曝光值
                    exposure_time_ok = new_exposure_times[directions_ok_keys[0]]
                    # 根据 keys 获取 boxstate
                    id2boxstate_ok = {int(i): id2boxstate[i] for i in directions_ok_keys}
                    exposure2boxes_ok = {
                        exposure_time_ok: id2boxstate_ok
                    }
                    exposure2id2boxstate.update(exposure2boxes_ok)

                if directions_high_keys.size > 0:
                    logger.info("adjust exposure high")
                    # 获取对应的曝光值
                    exposure_time_high = new_exposure_times[directions_high_keys[0]]
                    CameraConfig.setattr("exposure_time", exposure_time_high)
                    # 根据 keys 获取 boxstate
                    id2boxstate_high = {int(i): id2boxstate[i] for i in directions_high_keys}
                    exposure2boxes_high = adjust_exposure_full_res_recursive(camera_queue, id2boxstate_high)
                    exposure2id2boxstate.update(exposure2boxes_high)

    except queue.Empty:
        logger.error("get picture timeout")

    # 还原相机配置
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)
    #-------------------- 高分辨率快速拍摄 --------------------#

    # 去除 None box
    _exposure2id2boxstate = exposure2id2boxstate
    if id2boxstate is not None:
        _exposure2id2boxstate = {}
        for _exposure_time, _id2boxstate in exposure2id2boxstate.items():
            __id2boxstate = {}
            # id2boxstate: {0: {'ratio': 0.8184615384615387, 'score': 0.9265941381454468, 'box': [1509, 967, 1828, 1286]}}
            for i, boxstate in _id2boxstate.items():
                if boxstate['box'] is None:
                    continue
                __id2boxstate[i] = boxstate
            if len(__id2boxstate):
                _exposure2id2boxstate[_exposure_time] = __id2boxstate

    logger.success(f"{_exposure2id2boxstate = }")
    logger.info("adjust exposure end")
    return _exposure2id2boxstate


def adjust_exposure_full_res_for_loop(
    camera_queue: queue.Queue,
    id2boxstate: dict | None = None,
) -> dict[int, dict | None]:
    """使用循环实现快速调节曝光，全程使用高分辨率拍摄

    Args:
        camera_queue (queue.Queue): 相机队列
        id2boxstate (dict | None, optional): id对应box的状态，包括box和状态. Defaults to None.
            {
                i: {
                    "ratio": ratio,
                    "score": score,
                    "box": [x1, y1, x2, y2]
                },
                ...
            }

    Returns:
        dict[int, dict | None]: 曝光对应不同的box状态
            {
                exposure_time: {
                    i: {
                        "ratio": ratio,
                        "score": score,
                        "box": [x1, y1, x2, y2]
                    },
                    ...
                },
                ...
            }
    """
    logger.info("adjust exposure start")

    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
    adjust_total_times: int = AdjustCameraConfig.getattr("adjust_total_times")
    exposure2id2boxstate: dict[int, dict | None] = {}

    # 备份原本配置
    default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
    default_return_image_time_interval: int = CameraConfig.getattr("return_image_time_interval")
    # 快速拍照
    CameraConfig.setattr("capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval"))
    CameraConfig.setattr("return_image_time_interval", AdjustCameraConfig.getattr("return_image_time_interval"))

    # 使用栈来模拟递归
    stack = [(id2boxstate, CameraConfig.getattr("exposure_time"))]

    i = 0
    while stack:
        logger.info(f"stack size: {len(stack)}, i: {i}")
        current_id2boxstate, current_exposure_time = stack.pop()
        CameraConfig.setattr("exposure_time", current_exposure_time)

        # 超出次数, 设置为最后一次
        i += 1
        if i > adjust_total_times:
            logger.warning(f"adjust exposure times: {i}, final failed, set exposure time to {current_exposure_time} us")
            exposure2id2boxstate[current_exposure_time] = current_id2boxstate
            break

        try:
            # 获取图像
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

            # 全图调整
            if current_id2boxstate is None:
                new_exposure_time, direction = adjust_exposure_by_mean(
                    image,
                    image_metadata['ExposureTime'],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                # 可以
                if direction == 0:
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(f"full picture exposure time {new_exposure_time} us is ok")
                # 需要调整
                else:
                    logger.info(f"{new_exposure_time = }, {direction = }")
                    stack.append((None, new_exposure_time))

            # 划分box调整
            else:
                # directions 和 new_exposure_times 的 key 对应 boxid
                directions = {}
                new_exposure_times = {}
                for i, boxstate in current_id2boxstate.items():
                    box = boxstate["box"]
                    # 空box不处理
                    if box is None:
                        continue
                    target_image = image[box[1]:box[3], box[0]:box[2]]
                    # cv2.imwrite(f"./target_image_{i}.jpg", target_image)
                    new_exposure_time, direction = adjust_exposure_by_mean(
                        target_image,
                        image_metadata['ExposureTime'],
                        AdjustCameraConfig.getattr("mean_light_suitable_range"),
                        AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                        AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                    )

                    directions[i] = direction
                    new_exposure_times[i] = new_exposure_time
                    logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

                logger.info(f"{new_exposure_times = }")

                # 所有 box 都合适
                if all(direction == 0 for direction in directions.values()):
                    new_exposure_time = list(new_exposure_times.values())[0]
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(f"id2boxstate: {current_id2boxstate}, exposure time {new_exposure_time} us is ok")
                # box 需要分组
                else:
                    # exampe:
                    #   directions 和 new_exposure_times 的 key 对应 boxid
                    #   directions = { 1: 0, 3: 1, 4: -1, 5: 0, 7: 1}
                    #   new_exposure_times = { 1: 10, 3: 11, 4: 9, 5: 10, 7: 11}
                    #   directions_keys = [1, 3, 4, 5, 7]
                    #   directions_values = [0, 1, -1, 0, 1]
                    # 根据 directions_values 分为3组, 获取对应的index, 然后根据index获取keys, keys对应boxid

                    # 将box分组
                    # 根据 directions 判断曝光
                    directions_keys = np.array(list(directions.keys()))
                    directions_values = np.array(list(directions.values()))
                    # 获取 values 对应的 index
                    directions_low_index = np.where(directions_values==-1)[0]
                    directions_ok_index = np.where(directions_values==0)[0]
                    directions_high_index = np.where(directions_values==1)[0]
                    # 根据 index 获取 keys, keys 对应 boxid
                    directions_low_keys = directions_keys[directions_low_index]
                    directions_ok_keys = directions_keys[directions_ok_index]
                    directions_high_keys = directions_keys[directions_high_index]

                    # 分组
                    if directions_low_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_low = new_exposure_times[directions_low_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_low = {int(i): current_id2boxstate[i] for i in directions_low_keys}
                        # 放入栈中, 继续调整
                        stack.append((id2boxstate_low, exposure_time_low))
                        logger.success(f"id2boxstate: {id2boxstate_low}, exposure time need lower, adjust to {exposure_time_low} us")

                    if directions_ok_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_ok = new_exposure_times[directions_ok_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_ok = {int(i): current_id2boxstate[i] for i in directions_ok_keys}
                        # 放入最终结果
                        exposure2id2boxstate[exposure_time_ok] = id2boxstate_ok
                        logger.success(f"id2boxstate: {id2boxstate_ok}, exposure time {exposure_time_ok} us is ok")

                    if directions_high_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_high = new_exposure_times[directions_high_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_high = {int(i): current_id2boxstate[i] for i in directions_high_keys}
                        # 放入栈中, 继续调整
                        stack.append((id2boxstate_high, exposure_time_high))
                        logger.success(f"id2boxstate: {id2boxstate_high}, exposure time need heighter, adjust to {exposure_time_high} us")

        except queue.Empty:
            logger.error("get picture timeout")

    # 还原相机配置
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)

    # 去除 None box
    _exposure2id2boxstate = exposure2id2boxstate
    if id2boxstate is not None:
        _exposure2id2boxstate = {}
        for _exposure_time, _id2boxstate in exposure2id2boxstate.items():
            __id2boxstate = {}
            # id2boxstate: {0: {'ratio': 0.8184615384615387, 'score': 0.9265941381454468, 'box': [1509, 967, 1828, 1286]}}
            for i, boxstate in _id2boxstate.items():
                if boxstate['box'] is None:
                    continue
                __id2boxstate[i] = boxstate
            if len(__id2boxstate):
                _exposure2id2boxstate[_exposure_time] = __id2boxstate

    logger.success(f"{_exposure2id2boxstate = }")
    logger.info("adjust exposure end")
    return _exposure2id2boxstate


def adjust_exposure_low_res_for_loop(
    camera_queue: queue.Queue,
    id2boxstate: dict | None = None,
) -> dict[int, dict | None]:
    """使用循环实现快速调节曝光，全程使用低分辨率拍摄

    Args:
        camera_queue (queue.Queue): 相机队列
        id2boxstate (dict | None, optional): id对应box的状态，包括box和状态. Defaults to None.
            {
                i: {
                    "ratio": ratio,
                    "score": score,
                    "box": [x1, y1, x2, y2]
                },
                ...
            }

    Returns:
        dict[int, dict | None]: 曝光对应不同的box状态
            {
                exposure_time: {
                    i: {
                        "ratio": ratio,
                        "score": score,
                        "box": [x1, y1, x2, y2]
                    },
                    ...
                },
                ...
            }
    """
    logger.info("adjust exposure start")

    get_picture_timeout: int = MainConfig.getattr("get_picture_timeout")
    adjust_total_times: int = AdjustCameraConfig.getattr("adjust_total_times")
    exposure2id2boxstate: dict[int, dict | None] = {}

    # 备份原本配置
    default_capture_mode: str = CameraConfig.getattr("capture_mode")
    default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
    default_return_image_time_interval: int = CameraConfig.getattr("return_image_time_interval")

    # 找到低分辨率比率
    low_res_ratio: float = CameraConfig.getattr("low_res_ratio")

    # 设置快速拍照
    CameraConfig.setattr("capture_mode", AdjustCameraConfig.getattr("capture_mode"))
    CameraConfig.setattr("capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval"))
    CameraConfig.setattr("return_image_time_interval", AdjustCameraConfig.getattr("return_image_time_interval"))

    # 因为调整了分辨率, 因此需要清空队列
    clear_queue(camera_queue)

    # 使用栈来模拟递归
    stack = [(id2boxstate, CameraConfig.getattr("exposure_time"))]

    i = 0
    while stack:
        logger.info(f"stack size: {len(stack)}, i: {i}")
        current_id2boxstate, current_exposure_time = stack.pop()
        CameraConfig.setattr("exposure_time", current_exposure_time)

        # 超出次数, 设置为最后一次
        i += 1
        if i > adjust_total_times:
            logger.warning(f"adjust exposure times: {i}, final failed, set exposure time to {current_exposure_time} us")
            exposure2id2boxstate[current_exposure_time] = current_id2boxstate
            break

        try:
            # 获取图像
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

            # 全图调整
            if current_id2boxstate is None:
                new_exposure_time, direction = adjust_exposure_by_mean(
                    image,
                    image_metadata['ExposureTime'],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                # 可以
                if direction == 0:
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(f"full picture exposure time {new_exposure_time} us is ok")
                # 需要调整
                else:
                    logger.info(f"{new_exposure_time = }, {direction = }")
                    stack.append((None, new_exposure_time))

            # 划分box调整
            else:
                # directions 和 new_exposure_times 的 key 对应 boxid
                directions = {}
                new_exposure_times = {}
                for i, boxstate in current_id2boxstate.items():
                    box = boxstate["box"]
                    # 空box不处理
                    if box is None:
                        continue
                    # 调整 box 大小
                    _box: list[int] = [int(x * low_res_ratio) for x in box]
                    target_image = image[_box[1]:_box[3], _box[0]:_box[2]]
                    # cv2.imwrite(f"./target_image_{i}.jpg", target_image)
                    new_exposure_time, direction = adjust_exposure_by_mean(
                        target_image,
                        image_metadata['ExposureTime'],
                        AdjustCameraConfig.getattr("mean_light_suitable_range"),
                        AdjustCameraConfig.getattr("adjust_exposure_time_step"),
                        AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                    )

                    directions[i] = direction
                    new_exposure_times[i] = new_exposure_time
                    logger.info(f"boxid = {i}, {box = }, {new_exposure_time = }, {direction = }")

                logger.info(f"{new_exposure_times = }")

                # 所有 box 都合适
                if all(direction == 0 for direction in directions.values()):
                    new_exposure_time = list(new_exposure_times.values())[0]
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(f"id2boxstate: {current_id2boxstate}, exposure time {new_exposure_time} us is ok")
                # box 需要分组
                else:
                    # exampe:
                    #   directions 和 new_exposure_times 的 key 对应 boxid
                    #   directions = { 1: 0, 3: 1, 4: -1, 5: 0, 7: 1}
                    #   new_exposure_times = { 1: 10, 3: 11, 4: 9, 5: 10, 7: 11}
                    #   directions_keys = [1, 3, 4, 5, 7]
                    #   directions_values = [0, 1, -1, 0, 1]
                    # 根据 directions_values 分为3组, 获取对应的index, 然后根据index获取keys, keys对应boxid

                    # 将box分组
                    # 根据 directions 判断曝光
                    directions_keys = np.array(list(directions.keys()))
                    directions_values = np.array(list(directions.values()))
                    # 获取 values 对应的 index
                    directions_low_index = np.where(directions_values==-1)[0]
                    directions_ok_index = np.where(directions_values==0)[0]
                    directions_high_index = np.where(directions_values==1)[0]
                    # 根据 index 获取 keys, keys 对应 boxid
                    directions_low_keys = directions_keys[directions_low_index]
                    directions_ok_keys = directions_keys[directions_ok_index]
                    directions_high_keys = directions_keys[directions_high_index]

                    # 分组
                    if directions_low_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_low = new_exposure_times[directions_low_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_low = {int(i): current_id2boxstate[i] for i in directions_low_keys}
                        # 放入栈中, 继续调整
                        stack.append((id2boxstate_low, exposure_time_low))
                        logger.success(f"id2boxstate: {id2boxstate_low}, exposure time need lower, adjust to {exposure_time_low} us")

                    if directions_ok_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_ok = new_exposure_times[directions_ok_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_ok = {int(i): current_id2boxstate[i] for i in directions_ok_keys}
                        # 放入最终结果
                        exposure2id2boxstate[exposure_time_ok] = id2boxstate_ok
                        logger.success(f"id2boxstate: {id2boxstate_ok}, exposure time {exposure_time_ok} us is ok")

                    if directions_high_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_high = new_exposure_times[directions_high_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_high = {int(i): current_id2boxstate[i] for i in directions_high_keys}
                        # 放入栈中, 继续调整
                        stack.append((id2boxstate_high, exposure_time_high))
                        logger.success(f"id2boxstate: {id2boxstate_high}, exposure time need heighter, adjust to {exposure_time_high} us")

        except queue.Empty:
            logger.error("get picture timeout")

    # 还原相机配置
    CameraConfig.setattr("capture_mode", default_capture_mode)
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr("return_image_time_interval", default_return_image_time_interval)

    # 去除 None box
    _exposure2id2boxstate = exposure2id2boxstate
    if id2boxstate is not None:
        _exposure2id2boxstate = {}
        for _exposure_time, _id2boxstate in exposure2id2boxstate.items():
            __id2boxstate = {}
            # id2boxstate: {0: {'ratio': 0.8184615384615387, 'score': 0.9265941381454468, 'box': [1509, 967, 1828, 1286]}}
            for i, boxstate in _id2boxstate.items():
                if boxstate['box'] is None:
                    continue
                __id2boxstate[i] = boxstate
            if len(__id2boxstate):
                _exposure2id2boxstate[_exposure_time] = __id2boxstate

    logger.success(f"{_exposure2id2boxstate = }")
    logger.info("adjust exposure end")
    return _exposure2id2boxstate
