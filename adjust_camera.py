import numpy as np
from algorithm import mean_brightness
from config import MainConfig, CameraConfig, AdjustCameraConfig
import queue
from loguru import logger

from utils import clear_queue, drop_excessive_queue_items, get_picture_timeout_process


def adjust_exposure_by_mean(
    image: np.ndarray,
    exposure_time: float,
    mean_light_suitable_range: tuple[float, float],
    adjust_exposure_time_base_step: float = 100,
    suitable_ignore_ratio: float = 0.0,
) -> tuple[float, int]:
    """计算当前图像的平均亮度, 根据平均亮度测试是否需要调整曝光, 并调整曝光时间

    Args:
        image (np.ndarray): 图片
        exposure_time (float): 当前曝光时间
        mean_light_suitable_range (tuple[float, float]): 合适平均亮度范围
        adjust_exposure_time_base_step (float, optional): 调整曝光的步长. Defaults to 100.
        suitable_ignore_ratio (float, optional): 忽略合适的范围的比例, 0.1代表忽略最低 10% 和最高 10% 的范围. Defaults to 0.0.

    Returns:
        tuple[float, int]: 新的曝光值和调整方向
    """
    mean_bright = float(mean_brightness(image))

    # 动态 step, 根据距离合适的范围的中心距离调整
    suitable_range_middle = (
        mean_light_suitable_range[0] + mean_light_suitable_range[1]
    ) / 2
    step = int((suitable_range_middle - mean_bright) * adjust_exposure_time_base_step)

    # 缩小 mean_light_suitable_range 区间，让它更加宽松
    suitable_range = mean_light_suitable_range[1] - mean_light_suitable_range[0]
    ignore_range = suitable_range * suitable_ignore_ratio
    low = mean_light_suitable_range[0] + ignore_range
    high = mean_light_suitable_range[1] - ignore_range

    if mean_bright < low:
        logger.info(
            f"current_exposure_time = {exposure_time}, {mean_bright = }, {step = }, direction = 1"
        )
        return exposure_time + step, 1
    elif mean_bright > high:
        logger.info(
            f"current_exposure_time = {exposure_time}, {mean_bright = }, {step = }, direction = -1"
        )
        return exposure_time + step, -1
    else:
        logger.info(
            f"current_exposure_time = {exposure_time}, {mean_bright = }, direction = 0"
        )
        return exposure_time, 0


def adjust_exposure_by_mean_cuts(
    image: np.ndarray,
    boxes: list[list[int]],
    exposure_time: float,
    mean_light_suitable_range: tuple[float, float],
    adjust_exposure_time_base_step: float = 100,
    suitable_ignore_ratio: float = 0.0,
) -> tuple[float, int]:
    """先将图像切割为多个小图, 计算每个小图的平均亮度, 再计算所有小图的平均亮度, 根据平均亮度测试是否需要调整曝光, 并调整曝光时间

    Args:
        image (np.ndarray): 图片
        boxes (list[list[int]]): 图像切割的box
        exposure_time (float): 当前曝光时间
        mean_light_suitable_range (tuple[float, float]): 合适平均亮度范围
        adjust_exposure_time_base_step (float, optional): 调整曝光的步长. Defaults to 100.
        suitable_ignore_ratio (float, optional): 忽略合适的范围的比例, 0.1代表忽略最低 10% 和最高 10% 的范围. Defaults to 0.0.

    Returns:
        tuple[float, int]: 新的曝光值和调整方向
    """
    if len(boxes) == 0:
        return adjust_exposure_by_mean(
            image,
            exposure_time,
            mean_light_suitable_range,
            adjust_exposure_time_base_step,
            suitable_ignore_ratio,
        )

    mean_brights = []
    for box in boxes:
        cut_image = image[box[1] : box[3], box[0] : box[2]]
        mean_brights.append(mean_brightness(cut_image))

    # 计算所有小图的
    mean_bright = float(np.mean(mean_brights))

    # 动态 step, 根据距离合适的范围的中心距离调整
    suitable_range_middle = (
        mean_light_suitable_range[0] + mean_light_suitable_range[1]
    ) / 2
    step = int((suitable_range_middle - mean_bright) * adjust_exposure_time_base_step)

    # 缩小 mean_light_suitable_range 区间，让它更加宽松
    suitable_range = mean_light_suitable_range[1] - mean_light_suitable_range[0]
    ignore_range = suitable_range * suitable_ignore_ratio
    low = mean_light_suitable_range[0] + ignore_range
    high = mean_light_suitable_range[1] - ignore_range

    if mean_bright < low:
        logger.info(
            f"current_exposure_time = {exposure_time}, {mean_bright = }, {step = }, direction = 1"
        )
        return exposure_time + step, 1
    elif mean_bright > high:
        logger.info(
            f"current_exposure_time = {exposure_time}, {mean_bright = }, {step = }, direction = -1"
        )
        return exposure_time + step, -1
    else:
        logger.info(
            f"current_exposure_time = {exposure_time}, {mean_bright = }, direction = 0"
        )
        return exposure_time, 0


def adjust_exposure_full_res_for_loop(
    camera_queue: queue.Queue,
    id2boxstate: dict | None = None,
    cut_boxes_in_one_image: bool = False,
) -> tuple[dict[int, dict | None], bool, bool]:
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
        cut_boxes_in_one_image (bool, optional): 是否将box切割到一张图片, 而不是分别调整每个box. Defaults to False.

    Returns:
        tuple[dict[int, dict | None], bool, bool]: 曝光对应不同的box状态 和 需要更暗 和 需要更亮
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
    # low high 范围
    exposure_time_range: tuple[int, int] = AdjustCameraConfig.getattr(
        "exposure_time_range"
    )
    exposure2id2boxstate: dict[int, dict | None] = {}

    # 备份原本配置
    default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
    default_return_image_time_interval: int = CameraConfig.getattr(
        "return_image_time_interval"
    )
    # 快速拍照
    CameraConfig.setattr(
        "capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval")
    )
    CameraConfig.setattr(
        "return_image_time_interval",
        AdjustCameraConfig.getattr("return_image_time_interval"),
    )

    # 使用栈来模拟递归
    stack = [(id2boxstate, CameraConfig.getattr("exposure_time"))]

    # 是否需要闪光灯
    need_darker = False
    need_lighter = False

    i = 0
    while stack:
        i += 1
        logger.info(f"stack size: {len(stack)}, i: {i}")

        current_id2boxstate, current_exposure_time = stack.pop()
        CameraConfig.setattr("exposure_time", current_exposure_time)

        # 需要更暗
        if current_exposure_time < exposure_time_range[0]:
            logger.warning(
                f"exposure time: {current_exposure_time} us lower out of range, set exposure time to {exposure_time_range[0]} us"
            )
            exposure2id2boxstate[exposure_time_range[0]] = current_id2boxstate
            need_darker = True
            continue

        # 需要更亮
        if current_exposure_time > exposure_time_range[1]:
            logger.warning(
                f"exposure time: {current_exposure_time} us higher out of range, set exposure time to {exposure_time_range[1]} us, and need flash"
            )
            exposure2id2boxstate[exposure_time_range[1]] = current_id2boxstate
            need_lighter = True
            continue

        # 超出次数, 设置为最后一次
        if i > adjust_total_times:
            logger.warning(
                f"adjust exposure times: {i}, final failed, set exposure time to {current_exposure_time} us"
            )
            exposure2id2boxstate[current_exposure_time] = current_id2boxstate
            continue

        # 忽略多于图像
        drop_excessive_queue_items(camera_queue)

        try:
            image_timestamp, image, image_metadata = camera_queue.get(
                timeout=get_picture_timeout
            )
            logger.info(
                f"camera get image: {image_timestamp}, ExposureTime = {image_metadata['ExposureTime']}, AnalogueGain = {image_metadata['AnalogueGain']}, shape = {image.shape}"
            )

            if current_id2boxstate is None:
                # 全图调整
                logger.info("adjust exposure for full picture")
                new_exposure_time, direction = adjust_exposure_by_mean(
                    image,
                    image_metadata["ExposureTime"],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_base_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                # 可以
                if direction == 0:
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(
                        f"full picture exposure time {new_exposure_time} us is ok"
                    )
                # 需要调整
                else:
                    logger.info(f"{new_exposure_time = }, {direction = }")
                    stack.append((None, new_exposure_time))

            elif cut_boxes_in_one_image:
                # 全图调整, 一次性切割全部 box
                logger.info("adjust exposure for full picture with cut boxes")
                # 获取 boxes
                boxes = [
                    boxstate["box"]
                    for boxstate in current_id2boxstate.values()
                    if boxstate["box"] is not None
                ]
                new_exposure_time, direction = adjust_exposure_by_mean_cuts(
                    image,
                    boxes,
                    image_metadata["ExposureTime"],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_base_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                # 可以
                if direction == 0:
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(
                        f"full picture exposure time {new_exposure_time} us is ok"
                    )
                # 需要调整
                else:
                    logger.info(f"{new_exposure_time = }, {direction = }")
                    stack.append((current_id2boxstate, new_exposure_time))

            else:
                # 每个 box 单独调整
                logger.info("adjust exposure for every box")
                # directions 和 new_exposure_times 的 key 对应 boxid
                directions = {}
                new_exposure_times = {}
                for j, boxstate in current_id2boxstate.items():
                    box = boxstate["box"]
                    # 空box不处理
                    if box is None:
                        continue
                    target_image = image[box[1] : box[3], box[0] : box[2]]
                    # cv2.imwrite(f"./target_image_{j}.jpg", target_image)
                    new_exposure_time, direction = adjust_exposure_by_mean(
                        target_image,
                        image_metadata["ExposureTime"],
                        AdjustCameraConfig.getattr("mean_light_suitable_range"),
                        AdjustCameraConfig.getattr("adjust_exposure_time_base_step"),
                        AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                    )

                    directions[j] = direction
                    new_exposure_times[j] = new_exposure_time
                    logger.info(
                        f"boxid = {j}, {box = }, {new_exposure_time = }, {direction = }"
                    )

                logger.info(f"{directions = }")
                logger.info(f"{new_exposure_times = }")

                # 没有 box 需要调整
                if len(directions) == 0:
                    exposure2id2boxstate[current_exposure_time] = current_id2boxstate
                    logger.warning("no box can adjust, use current exposure time")
                # 所有 box 都合适
                elif all(direction == 0 for direction in directions.values()):
                    new_exposure_time = list(new_exposure_times.values())[0]
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(
                        f"id2boxstate: {current_id2boxstate}, exposure time {new_exposure_time} us is ok"
                    )
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
                    directions_low_index = np.where(directions_values == -1)[0]
                    directions_ok_index = np.where(directions_values == 0)[0]
                    directions_high_index = np.where(directions_values == 1)[0]
                    # 根据 index 获取 keys, keys 对应 boxid
                    directions_low_keys = directions_keys[directions_low_index]
                    directions_ok_keys = directions_keys[directions_ok_index]
                    directions_high_keys = directions_keys[directions_high_index]

                    # 分组
                    if directions_low_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_low = new_exposure_times[directions_low_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_low = {
                            int(i): current_id2boxstate[i] for i in directions_low_keys
                        }
                        # 放入栈中, 继续调整
                        stack.append((id2boxstate_low, exposure_time_low))
                        logger.success(
                            f"id2boxstate: {id2boxstate_low}, exposure time need lower, adjust to {exposure_time_low} us"
                        )

                    if directions_ok_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_ok = new_exposure_times[directions_ok_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_ok = {
                            int(i): current_id2boxstate[i] for i in directions_ok_keys
                        }
                        # 放入最终结果
                        exposure2id2boxstate[exposure_time_ok] = id2boxstate_ok
                        logger.success(
                            f"id2boxstate: {id2boxstate_ok}, exposure time {exposure_time_ok} us is ok"
                        )

                    if directions_high_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_high = new_exposure_times[directions_high_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_high = {
                            int(i): current_id2boxstate[i] for i in directions_high_keys
                        }
                        # 放入栈中, 继续调整
                        stack.append((id2boxstate_high, exposure_time_high))
                        logger.success(
                            f"id2boxstate: {id2boxstate_high}, exposure time need heighter, adjust to {exposure_time_high} us"
                        )

        except queue.Empty:
            get_picture_timeout_process()

    # 还原相机配置
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr(
        "return_image_time_interval", default_return_image_time_interval
    )

    return exposure2id2boxstate, need_darker, need_lighter

    # 去除 None box
    _exposure2id2boxstate = exposure2id2boxstate
    if id2boxstate is not None:
        _exposure2id2boxstate = {}
        for _exposure_time, _id2boxstate in exposure2id2boxstate.items():
            __id2boxstate = {}
            # id2boxstate: {0: {'ratio': 0.8184615384615387, 'score': 0.9265941381454468, 'box': [1509, 967, 1828, 1286]}}
            for i, boxstate in _id2boxstate.items():
                if boxstate["box"] is None:
                    continue
                __id2boxstate[i] = boxstate
            if len(__id2boxstate):
                _exposure2id2boxstate[_exposure_time] = __id2boxstate

    logger.success(f"{_exposure2id2boxstate = }")
    logger.info("adjust exposure end")
    return _exposure2id2boxstate, need_darker, need_lighter


def adjust_exposure_low_res_for_loop(
    camera_queue: queue.Queue,
    id2boxstate: dict | None = None,
    cut_boxes_in_one_image: bool = False,
) -> tuple[dict[int, dict | None], bool, bool]:
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
        cut_boxes_in_one_image (bool, optional): 是否将box切割到一张图片, 而不是分别调整每个box. Defaults to False.

    Returns:
        tuple[dict[int, dict | None], bool, bool]: 曝光对应不同的box状态 和 需要更暗 和 需要更亮
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
    # low high 范围
    exposure_time_range: tuple[int, int] = AdjustCameraConfig.getattr(
        "exposure_time_range"
    )
    exposure2id2boxstate: dict[int, dict | None] = {}

    # 备份原本配置
    default_capture_mode: str = CameraConfig.getattr("capture_mode")
    default_capture_time_interval: int = CameraConfig.getattr("capture_time_interval")
    default_return_image_time_interval: int = CameraConfig.getattr(
        "return_image_time_interval"
    )

    # 找到低分辨率比率
    low_res_ratio: float = CameraConfig.getattr("low_res_ratio")

    # 设置快速拍照
    CameraConfig.setattr("capture_mode", AdjustCameraConfig.getattr("capture_mode"))
    CameraConfig.setattr(
        "capture_time_interval", AdjustCameraConfig.getattr("capture_time_interval")
    )
    CameraConfig.setattr(
        "return_image_time_interval",
        AdjustCameraConfig.getattr("return_image_time_interval"),
    )

    # 因为调整了分辨率, 因此需要清空队列
    clear_queue(camera_queue)

    # 使用栈来模拟递归
    stack = [(id2boxstate, CameraConfig.getattr("exposure_time"))]

    # 是否需要闪光灯
    need_darker = False
    need_lighter = False

    i = 0
    while stack:
        i += 1
        logger.info(f"stack size: {len(stack)}, i: {i}")

        current_id2boxstate, current_exposure_time = stack.pop()
        CameraConfig.setattr("exposure_time", current_exposure_time)

        # 需要更暗
        if current_exposure_time < exposure_time_range[0]:
            logger.warning(
                f"exposure time: {current_exposure_time} us lower out of range, set exposure time to {exposure_time_range[0]} us"
            )
            exposure2id2boxstate[exposure_time_range[0]] = current_id2boxstate
            need_darker = True
            continue

        # 需要更亮
        if current_exposure_time > exposure_time_range[1]:
            logger.warning(
                f"exposure time: {current_exposure_time} us higher out of range, set exposure time to {exposure_time_range[1]} us, and need flash"
            )
            exposure2id2boxstate[exposure_time_range[1]] = current_id2boxstate
            need_lighter = True
            continue

        # 超出次数, 设置为最后一次
        if i > adjust_total_times:
            logger.warning(
                f"adjust exposure times: {i}, final failed, set exposure time to {current_exposure_time} us"
            )
            exposure2id2boxstate[current_exposure_time] = current_id2boxstate
            continue

        # 忽略多于图像
        drop_excessive_queue_items(camera_queue)

        try:
            image_timestamp, image, image_metadata = camera_queue.get(
                timeout=get_picture_timeout
            )
            logger.info(
                f"camera get image: {image_timestamp}, ExposureTime = {image_metadata['ExposureTime']}, AnalogueGain = {image_metadata['AnalogueGain']}, shape = {image.shape}"
            )

            if current_id2boxstate is None:
                # 全图调整
                logger.info("adjust exposure for full picture")
                new_exposure_time, direction = adjust_exposure_by_mean(
                    image,
                    image_metadata["ExposureTime"],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_base_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                # 可以
                if direction == 0:
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(
                        f"full picture exposure time {new_exposure_time} us is ok"
                    )
                # 需要调整
                else:
                    logger.info(f"{new_exposure_time = }, {direction = }")
                    stack.append((None, new_exposure_time))

            elif cut_boxes_in_one_image:
                # 全图调整, 一次性切割全部 box
                logger.info("adjust exposure for full picture with cut boxes")
                # 获取 boxes
                boxes = [
                    boxstate["box"]
                    for boxstate in current_id2boxstate.values()
                    if boxstate["box"] is not None
                ]
                # 调整 box 大小
                boxes = [[int(b * low_res_ratio) for b in box] for box in boxes]
                new_exposure_time, direction = adjust_exposure_by_mean_cuts(
                    image,
                    boxes,
                    image_metadata["ExposureTime"],
                    AdjustCameraConfig.getattr("mean_light_suitable_range"),
                    AdjustCameraConfig.getattr("adjust_exposure_time_base_step"),
                    AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                )

                # 可以
                if direction == 0:
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(
                        f"full picture exposure time {new_exposure_time} us is ok"
                    )
                # 需要调整
                else:
                    logger.info(f"{new_exposure_time = }, {direction = }")
                    stack.append((current_id2boxstate, new_exposure_time))

            else:
                # 每个 box 单独调整
                logger.info("adjust exposure for every box")
                # directions 和 new_exposure_times 的 key 对应 boxid
                directions = {}
                new_exposure_times = {}
                for j, boxstate in current_id2boxstate.items():
                    box = boxstate["box"]
                    # 空box不处理
                    if box is None:
                        continue
                    # 调整 box 大小
                    _box: list[int] = [int(x * low_res_ratio) for x in box]
                    target_image = image[_box[1] : _box[3], _box[0] : _box[2]]
                    # cv2.imwrite(f"./target_image_{j}.jpg", target_image)
                    new_exposure_time, direction = adjust_exposure_by_mean(
                        target_image,
                        image_metadata["ExposureTime"],
                        AdjustCameraConfig.getattr("mean_light_suitable_range"),
                        AdjustCameraConfig.getattr("adjust_exposure_time_base_step"),
                        AdjustCameraConfig.getattr("suitable_ignore_ratio"),
                    )

                    directions[j] = direction
                    new_exposure_times[j] = new_exposure_time
                    logger.info(
                        f"boxid = {j}, {box = }, {new_exposure_time = }, {direction = }"
                    )

                logger.info(f"{new_exposure_times = }")

                # 没有 box 需要调整
                if len(directions) == 0:
                    exposure2id2boxstate[current_exposure_time] = current_id2boxstate
                    logger.warning("no box can adjust, use current exposure time")
                # 所有 box 都合适
                elif all(direction == 0 for direction in directions.values()):
                    new_exposure_time = list(new_exposure_times.values())[0]
                    exposure2id2boxstate[new_exposure_time] = current_id2boxstate
                    logger.success(
                        f"id2boxstate: {current_id2boxstate}, exposure time {new_exposure_time} us is ok"
                    )
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
                    directions_low_index = np.where(directions_values == -1)[0]
                    directions_ok_index = np.where(directions_values == 0)[0]
                    directions_high_index = np.where(directions_values == 1)[0]
                    # 根据 index 获取 keys, keys 对应 boxid
                    directions_low_keys = directions_keys[directions_low_index]
                    directions_ok_keys = directions_keys[directions_ok_index]
                    directions_high_keys = directions_keys[directions_high_index]

                    # 分组
                    if directions_low_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_low = new_exposure_times[directions_low_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_low = {
                            int(i): current_id2boxstate[i] for i in directions_low_keys
                        }
                        # 放入栈中, 继续调整
                        stack.append((id2boxstate_low, exposure_time_low))
                        logger.success(
                            f"id2boxstate: {id2boxstate_low}, exposure time need lower, adjust to {exposure_time_low} us"
                        )

                    if directions_ok_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_ok = new_exposure_times[directions_ok_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_ok = {
                            int(i): current_id2boxstate[i] for i in directions_ok_keys
                        }
                        # 放入最终结果
                        exposure2id2boxstate[exposure_time_ok] = id2boxstate_ok
                        logger.success(
                            f"id2boxstate: {id2boxstate_ok}, exposure time {exposure_time_ok} us is ok"
                        )

                    if directions_high_keys.size > 0:
                        # 获取对应的曝光值
                        exposure_time_high = new_exposure_times[directions_high_keys[0]]
                        # 根据 keys 获取 boxstate
                        id2boxstate_high = {
                            int(i): current_id2boxstate[i] for i in directions_high_keys
                        }
                        # 放入栈中, 继续调整
                        stack.append((id2boxstate_high, exposure_time_high))
                        logger.success(
                            f"id2boxstate: {id2boxstate_high}, exposure time need heighter, adjust to {exposure_time_high} us"
                        )

        except queue.Empty:
            get_picture_timeout_process()

    # 还原相机配置
    CameraConfig.setattr("capture_mode", default_capture_mode)
    CameraConfig.setattr("capture_time_interval", default_capture_time_interval)
    CameraConfig.setattr(
        "return_image_time_interval", default_return_image_time_interval
    )

    return exposure2id2boxstate, need_darker, need_lighter

    # 去除 None box
    _exposure2id2boxstate = exposure2id2boxstate
    if id2boxstate is not None:
        _exposure2id2boxstate = {}
        for _exposure_time, _id2boxstate in exposure2id2boxstate.items():
            __id2boxstate = {}
            # id2boxstate: {0: {'ratio': 0.8184615384615387, 'score': 0.9265941381454468, 'box': [1509, 967, 1828, 1286]}}
            for i, boxstate in _id2boxstate.items():
                if boxstate["box"] is None:
                    continue
                __id2boxstate[i] = boxstate
            if len(__id2boxstate):
                _exposure2id2boxstate[_exposure_time] = __id2boxstate

    logger.success(f"{_exposure2id2boxstate = }")
    logger.info("adjust exposure end")
    return _exposure2id2boxstate, need_darker, need_lighter
