import numpy as np
from numba import jit
import cv2
import matplotlib.pyplot as plt
from typing import Literal
from loguru import logger


def box_iou(box1: list, box2: list) -> float:
    """calc iou

    Args:
        box1 (list): 盒子1 [x_min, y_min, x_max, y_max]
        box2 (list): 盒子2 [x_min, y_min, x_max, y_max]

    Returns:
        iou (float)
    """
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 内部盒子面积
    inner_box_x1 = max(box1[0], box2[0])
    inner_box_y1 = max(box1[1], box2[1])
    inner_box_x2 = min(box1[2], box2[2])
    inner_box_y2 = min(box1[3], box2[3])
    # max 用来判断是否重叠
    inner_box_area = max(inner_box_x2 - inner_box_x1, 0) * max(inner_box_y2 - inner_box_y1, 0)

    iou = inner_box_area / (box1_area + box2_area - inner_box_area)
    return iou


def test_box_iou():
    print(box_iou([0, 0, 1, 1], [0, 0, 1, 1]))  # 1.0
    print(box_iou([1, 1, 2, 2], [0, 0, 3, 3]))  # 0.1111111111111111
    print(box_iou([0, 0, 4, 4], [0, 0, 2, 2]))  # 0.25
    print(box_iou([0, 0, 1, 1], [1, 1, 2, 2]))  # 0.0


def sort_boxes(boxes: np.ndarray, sort_by: Literal["x", "y", "xy"] = "x") -> np.ndarray:
    """根据 x or y 从小到大排序对 boxes 进行排序

    Args:
        boxes (np.ndarray): 未排序的box,  [[x_min, y_min, x_max, y_max],...]
        sort_by (Literal["x", "y", "xy"], optional): 排序方式. Defaults to "x".

    Returns:
        np.ndarray: 排序的box
    """
    assert sort_by in ["x", "y", "xy"], f"sort_by must be 'x', 'y' or 'xy' but got {sort_by}"

    # lexsort: 给定多个排序键，lexsort返回一个整数索引数组，该数组按多个键描述排序顺序。序列中的最后一个键用于主排序，倒数第二个键用于次排序，以此类推。
    if sort_by == "x":
        combined_indices = np.lexsort((boxes[:, 3], boxes[:, 1], boxes[:, 2], boxes[:, 0]))
    elif sort_by == "y":
        combined_indices = np.lexsort((boxes[:, 2], boxes[:, 0], boxes[:, 3], boxes[:, 1]))
    else:
        combined_indices = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))

    return boxes[combined_indices]


def test_sort_boxes():
    boxes = np.array([[0, 4, 4, 5], [0, 0, 5, 4], [6, 6, 10, 11], [1, 6, 7, 8], [6, 4, 9, 7], [1, 8, 12, 10]])
    print(sort_boxes(boxes))
    # [[ 0  4  4  5]
    #  [ 0  0  5  4]
    #  [ 1  6  7  8]
    #  [ 1  8 12 10]
    #  [ 6  4  9  7]
    #  [ 6  6 10 11]]
    print(sort_boxes(boxes, "y"))
    # [[ 0  0  5  4]
    #  [ 0  4  4  5]
    #  [ 6  4  9  7]
    #  [ 1  6  7  8]
    #  [ 6  6 10 11]
    #  [ 1  8 12 10]]
    print(sort_boxes(boxes, "xy"))
    # [[ 0  0  5  4]
    #  [ 0  4  4  5]
    #  [ 1  6  7  8]
    #  [ 1  8 12 10]
    #  [ 6  4  9  7]
    #  [ 6  6 10 11]]
    print()


def sort_boxes_center(boxes: np.ndarray, sort_by: Literal["x", "y"] = "x") -> np.ndarray:
    """根据中心 x or y 从小到大排序对 boxes 进行排序
    """
    assert sort_by in ["x", "y"], f"sort_by must be 'x' or 'y' but got {sort_by}"

    x_center = (boxes[:, 0] + boxes[:, 2]) / 2
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2

    # lexsort: 给定多个排序键，lexsort返回一个整数索引数组，该数组按多个键描述排序顺序。序列中的最后一个键用于主排序，倒数第二个键用于次排序，以此类推。
    if sort_by == "x":
        combined_indices = np.lexsort((y_center, x_center))
    else:
        combined_indices = np.lexsort((x_center, y_center))
    return boxes[combined_indices]


def test_sort_boxes_center():
    boxes = np.array([[0, 4, 4, 5], [0, 0, 5, 4], [6, 6, 10, 11], [1, 6, 7, 8], [6, 4, 9, 7], [1, 8, 12, 10]])
    print(sort_boxes_center(boxes))
    # [[ 0  4  4  5]
    #  [ 0  0  5  4]
    #  [ 1  6  7  8]
    #  [ 1  8 12 10]
    #  [ 6  4  9  7]
    #  [ 6  6 10 11]]
    print(sort_boxes_center(boxes, "y"))
    # [[ 0  0  5  4]
    #  [ 0  4  4  5]
    #  [ 6  4  9  7]
    #  [ 1  6  7  8]
    #  [ 6  6 10 11]
    #  [ 1  8 12 10]]


def iou_filter_by_threshold(boxes: list | np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """iou阈值过滤

    Args:
        boxes (list | np.ndarray): [[x_min, y_min, x_max, y_max],...]
        iou_threshold (float, optional): iou threshold. Defaults to 0.5.

    Returns:
        np.ndarray: 保留的index
    """
    boxes_len = len(boxes)
    # 根据 iou 过滤
    keep_bool_index = np.full((boxes_len,), True)
    for i in range(0, boxes_len - 1):
        box1 = boxes[i]
        if keep_bool_index[i] == False:
            continue
        for j in range(i + 1, boxes_len):
            if keep_bool_index[j] == False:
                continue
            box2 = boxes[j]
            iou = box_iou(box1, box2)
            if iou > iou_threshold:
                keep_bool_index[j] = False

    # 获取保留的 index
    reserve_index = np.where(keep_bool_index)[0]
    return reserve_index


def match_template_max(
    image: np.ndarray,
    template: np.ndarray,
    match_method: int = cv2.TM_CCOEFF_NORMED,
) -> list[tuple[float, list[int]]]:
    """模板匹配，获取匹配阈值最大/最小的位置

    Args:
        image (np.ndarray): 图片
        template (np.ndarray): 模板图片
        match_method (int, optional): 匹配算法. Defaults to cv2.TM_SQDIFF_NORMED.
            TM_SQDIFF:        方差匹配方法, 越小代表越准确
            TM_SQDIFF_NORMED: 归一化的方差匹配方法, 越小代表越准确
            TM_CCORR:         相关性匹配方法, 越大代表越准确
            TM_CCORR_NORMED:  归一化的相关性匹配方法, 越大代表越准确, 对亮度变化不敏感
            TM_CCOEFF:        相关系数匹配方法, 越大代表越准确
            TM_CCOEFF_NORMED: 归一化的相关系数匹配方法, 越大代表越准确, 对亮度变化不敏感

    Returns:
        list[tuple[float, list[int]]]: [(最高得分和框的坐标, [x1, y1, x2, y2])]
    """
    template_h, template_w = template.shape[:2]

    match_image = cv2.matchTemplate(
        image=image,        # 图片
        templ=template,     # 模板
        method=match_method,# 匹配方法
    )

    # min_loc: [x, y]
    # max_loc: [x, y]
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(src=match_image)

    # 在函数完成比较后，可以使用minMaxLoc函数将最佳匹配作为全局最小值（当使用TM_SQDIFF时）或最大值（当采用TM_CCORR或TM_CCOEFF时）。
    # 在彩色图像的情况下，在所有通道上进行分子中的模板求和和和分母中的每个求和，并且对每个通道使用单独的平均值。
    # 也就是说，该函数可以采用颜色模板和彩色图像。结果仍然是单通道图像，更易于分析。
    if match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED:
        score = min_value
        real_loc = min_loc
    else:
        score = max_value
        real_loc = max_loc

    box = [*real_loc, real_loc[0] + template_w, real_loc[1] + template_h]
    return [(score, box)]


def match_template_filter_by_threshold(
    image: np.ndarray,
    template: np.ndarray,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    match_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> list[tuple[float, list[int]]]:
    """模板匹配根据匹配阈值过滤，配合 iou 过滤

    Args:
        image (np.ndarray): 图片
        template (np.ndarray): 模板图片
        match_method (int, optional): 匹配算法. Defaults to cv2.TM_SQDIFF_NORMED.
            TM_SQDIFF:        方差匹配方法, 越小代表越准确
            TM_SQDIFF_NORMED: 归一化的方差匹配方法, 越小代表越准确
            TM_CCORR:         相关性匹配方法, 越大代表越准确
            TM_CCORR_NORMED:  归一化的相关性匹配方法, 越大代表越准确, 对亮度变化不敏感
            TM_CCOEFF:        相关系数匹配方法, 越大代表越准确
            TM_CCOEFF_NORMED: 归一化的相关系数匹配方法, 越大代表越准确, 对亮度变化不敏感
        match_threshold (float, optional): 匹配阈值. Defaults to 0.5.
        iou_threshold (float, optional): iou threshold. Defaults to 0.5.

    Returns:
        list[tuple[float, list[int]]]:
    """

    template_h, template_w = template.shape[:2]

    match_image = cv2.matchTemplate(
        image=image,        # 图片
        templ=template,     # 模板
        method=match_method,# 匹配方法
    )

    # 在函数完成比较后，可以使用minMaxLoc函数将最佳匹配作为全局最小值（当使用TM_SQDIFF时）或最大值（当采用TM_CCORR或TM_CCOEFF时）。
    # 在彩色图像的情况下，在所有通道上进行分子中的模板求和和和分母中的每个求和，并且对每个通道使用单独的平均值。
    # 也就是说，该函数可以采用颜色模板和彩色图像。结果仍然是单通道图像，更易于分析。
    if match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED:
        keep_y, keep_x = np.where(match_image < match_threshold)
        scores = match_image[keep_y, keep_x]
        # sort scores, 升序
        sort_index = np.argsort(scores)
    else:
        keep_y, keep_x = np.where(match_image > match_threshold)
        scores = match_image[keep_y, keep_x]
        # sort scores, 降序
        sort_index = np.argsort(-scores)

    # [[x1, y1, x2, y2]...]
    boxes = np.array([keep_x, keep_y, keep_x + template_w, keep_y + template_h]).T
    # 按照得分排序
    scores = scores[sort_index]
    boxes = boxes[sort_index]
    # logger.info(f"{boxes.shape = }")

    # 根据iou过滤
    reserve_index = iou_filter_by_threshold(boxes, iou_threshold)
    reserve_scores = scores[reserve_index]
    reserve_boxes = boxes[reserve_index]
    # logger.info(f"{reserve_boxes.shape = }")

    match_result = list(zip(reserve_scores, reserve_boxes))
    return match_result


def multi_target_match_template(
    image: np.ndarray,
    template: np.ndarray,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    target_number: int = 1,
) -> tuple[float, list[int]]:
    """模板匹配多个目标, 匹配方法为将每次匹配到的目标屏蔽掉, 进行多次匹配

    Args:
        image (np.ndarray): 图片
        template (np.ndarray): 模板图片
        match_method (int, optional): 匹配算法. Defaults to cv2.TM_SQDIFF_NORMED.
            TM_SQDIFF:        方差匹配方法, 越小代表越准确
            TM_SQDIFF_NORMED: 归一化的方差匹配方法, 越小代表越准确
            TM_CCORR:         相关性匹配方法, 越大代表越准确
            TM_CCORR_NORMED:  归一化的相关性匹配方法, 越大代表越准确, 对亮度变化不敏感
            TM_CCOEFF:        相关系数匹配方法, 越大代表越准确
            TM_CCOEFF_NORMED: 归一化的相关系数匹配方法, 越大代表越准确, 对亮度变化不敏感
        target_number (int, optional): 匹配目标数量. Defaults to 1.
    Returns:
        tuple[float, list[list[int]]: 最高得分和框的坐标, [x1, y1, x2, y2]
    """
    image_ = image.copy()
    results = []
    for i in range(target_number):
        match_result = match_template_max(image_, template, match_method)
        score, box = match_result[0]
        results.append([score, box])
        if i != target_number - 1:
            # 将找到的区域设置为随机像素
            image_[box[1]:box[3], box[0]:box[2]] = np.random.randint(0, 255, (box[3] - box[1], box[2] - box[0]))
    return results


def multi_scale_match_template(
    image: np.ndarray,
    template: np.ndarray,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    init_scale: float = 0.125,
    scales: tuple[float] = (1.0, 4.0, 0.1),
    use_threshold_match: bool = True,
    threshold_match_threshold: float = 0.8,
    threshold_iou_threshold: float = 0.5,
) -> list:
    """多尺度模板匹配
    先根据 init_scale 将模板调整为相对于图片的大小，然后再循环 scales 调整 template 的大小进行匹配

    Args:
        image (np.ndarray): 图片
        template (np.ndarray): 模板图片
        match_method (int, optional): 匹配方法. Defaults to cv2.TM_CCOEFF_NORMED.
            TM_SQDIFF:        方差匹配方法, 越小代表越准确
            TM_SQDIFF_NORMED: 归一化的方差匹配方法, 越小代表越准确
            TM_CCORR:         相关性匹配方法, 越大代表越准确
            TM_CCORR_NORMED:  归一化的相关性匹配方法, 越大代表越准确, 对亮度变化不敏感
            TM_CCOEFF:        相关系数匹配方法, 越大代表越准确
            TM_CCOEFF_NORMED: 归一化的相关系数匹配方法, 越大代表越准确, 对亮度变化不敏感
        init_scale (float, optional): 将模板的最小边长调整为图片最小边长的比例. Defaults to 0.125.
        scales (tuple[float], optional): 缩放的范围, (start, end, step), include end. Defaults to (1.0, 4.0, 0.1).
        use_threshold_match (bool, optional): 是否使用阈值匹配. Defaults to True.
        threshold_match_threshold (float, optional): 匹配阈值. Defaults to 0.8.
        threshold_iou_threshold (float, optional): iou threshold. Defaults to 0.5.

    Returns:
        list: 每个 scale 匹配的模板结果, (scale, score, box)
    """

    # 将模板大小调整到相对于图片合适的大小
    template_h, template_w = template.shape[:2]
    image_h, image_w = image.shape[:2]
    ratio = min(image_h, image_w) * init_scale / min(template_h, template_w)
    # 临时尺寸
    _template_h = template_h * ratio
    _template_w = template_w * ratio

    # 模糊图像
    # image = cv2.GaussianBlur(image, (3, 3), 0)

    match_results = []
    # 根据不同尺度调整模板的大小
    for scale in np.arange(scales[0], scales[1] + 1e-8, scales[2]).tolist():
        # 临时尺寸 * scale 得到最终尺寸
        resized_h = int(_template_h * scale)
        resized_w = int(_template_w * scale)
        # 使用最终尺寸一次 resize 模板
        template_resized = cv2.resize(template, (resized_w, resized_h))
        logger.info(f"{scale = }, resize template size h = {resized_h}, w = {resized_w}")
        # 模糊模板
        # template_resized = cv2.GaussianBlur(template_resized, (3, 3), 0)

        if not use_threshold_match:
            # 选择最高得分
            match_result = match_template_max(image, template_resized, match_method)
            score, box = match_result[0] # 最高得分
            match_results.append((scale, score, box))
        else:
            # 通过阈值匹配
            match_result = match_template_filter_by_threshold(
                image,
                template_resized,
                match_method,
                threshold_match_threshold,
                threshold_iou_threshold,
            )
            # scale: int
            # match_result: [[score, box]...]  -> [[scale, score, box]...]
            _match_result = list(zip([scale] * len(match_result), *list(zip(*match_result))))
            match_results.extend(_match_result)

    if match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED:
        reverse = False
    else:
        reverse = True
    # 按照得分排序
    match_results = sorted(match_results, key=lambda x: x[1], reverse=reverse)
    # logger.info(f"match_results: {match_results}")
    logger.info(f"match_results number: {len(match_results)}")
    return match_results


def multi_target_multi_scale_match_template_old(
    image: np.ndarray,
    template: np.ndarray,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    init_scale: float = 0.125,
    scales: tuple[float] = (1.0, 4.0, 0.1),
    target_number: int = 1,
) -> list:
    """多目标多尺度匹配,将每次匹配到的位置屏蔽掉，然后重新匹配
    先根据 init_scale 将模板调整为相对于图片的大小，然后再循环 scales 调整 template 的大小进行匹配

    Args:
        image (np.ndarray): 图片
        template (np.ndarray): 模板图片
        match_method (int, optional): 匹配方法. Defaults to cv2.TM_CCOEFF_NORMED.
            TM_SQDIFF:        方差匹配方法, 越小代表越准确
            TM_SQDIFF_NORMED: 归一化的方差匹配方法, 越小代表越准确
            TM_CCORR:         相关性匹配方法, 越大代表越准确
            TM_CCORR_NORMED:  归一化的相关性匹配方法, 越大代表越准确, 对亮度变化不敏感
            TM_CCOEFF:        相关系数匹配方法, 越大代表越准确
            TM_CCOEFF_NORMED: 归一化的相关系数匹配方法, 越大代表越准确, 对亮度变化不敏感
        init_scale (float, optional): 将模板的最小边长调整为图片最小边长的比例. Defaults to 0.125.
        scales (tuple[float], optional): 缩放的范围, (start, end, step), include end. Defaults to (1.0, 4.0, 0.1).
        target_number (int, optional): 匹配目标数量. Defaults to 1.

    Returns:
        list: 多目标匹配结果 boxes
    """
    image_ = image.copy()
    boxes = []
    for i in range(target_number):
        logger.info(f"match number {i + 1} target")
        result = multi_scale_match_template(
            image_,
            template,
            match_method,
            init_scale,
            scales
        )
        best_result = result[0] # 选择最好的匹配结果
        logger.info(f"match number {i + 1} target best result: {best_result}")
        box = best_result[2]
        boxes.append(box)
        if i != target_number - 1:
            # 将找到的区域设置为随机像素
            image_[box[1]:box[3], box[0]:box[2]] = np.random.randint(0, 255, (box[3] - box[1], box[2] - box[0]))

    return boxes


def multi_target_multi_scale_match_template(
    image: np.ndarray,
    template: np.ndarray,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    init_scale: float = 0.125,
    scales: tuple[float] = (1.0, 4.0, 0.1),
    target_number: int = -1,
    iou_threshold: float = 0.5,
    use_threshold_match: bool = True,
    threshold_match_threshold: float = 0.8,
    threshold_iou_threshold: float = 0.5,
) -> np.ndarray:
    """多目标多尺度匹配
    先根据 init_scale 将模板调整为相对于图片的大小，然后再循环 scales 调整 template 的大小进行匹配

    Args:
        image (np.ndarray): 图片
        template (np.ndarray): 模板图片
        match_method (int, optional): 匹配方法. Defaults to cv2.TM_CCOEFF_NORMED.
            TM_SQDIFF:        方差匹配方法, 越小代表越准确
            TM_SQDIFF_NORMED: 归一化的方差匹配方法, 越小代表越准确
            TM_CCORR:         相关性匹配方法, 越大代表越准确
            TM_CCORR_NORMED:  归一化的相关性匹配方法, 越大代表越准确, 对亮度变化不敏感
            TM_CCOEFF:        相关系数匹配方法, 越大代表越准确
            TM_CCOEFF_NORMED: 归一化的相关系数匹配方法, 越大代表越准确, 对亮度变化不敏感
        init_scale (float, optional): 将模板的最小边长调整为图片最小边长的比例. Defaults to 0.125.
        scales (tuple[float], optional): 缩放的范围, (start, end, step), include end. Defaults to (1.0, 4.0, 0.1).
        target_number (int, optional): 匹配目标数量, -1 表示不限制. Defaults to -1.
        iou_threshold (float, optional): 重叠框的 iou 阈值. Defaults to 0.5.
        use_threshold_match (bool, optional): 是否使用阈值匹配. Defaults to True.
        threshold_match_threshold (float, optional): 匹配阈值. Defaults to 0.8.
        threshold_iou_threshold (float, optional): iou threshold. Defaults to 0.5.

    Returns:
        np.ndarray: 多目标匹配结果 boxes, [[x_min, y_min, x_max, y_max],...]
    """
    # 多目标匹配
    results = multi_scale_match_template(
        image,
        template,
        match_method,
        init_scale,
        scales,
        use_threshold_match,
        threshold_match_threshold,
        threshold_iou_threshold,
    )
    # logger.info(f"multi_scale_match_template results: {results}")
    _scales = np.array([result[0] for result in results])
    logger.info(f"scales:\n {_scales}")
    scores = np.array([result[1] for result in results])
    logger.info(f"scores:\n {scores}")
    boxes = np.array([result[2] for result in results])
    logger.info(f"boxes:\n {boxes}")

    # iou 过滤
    reserve_index = iou_filter_by_threshold(boxes, iou_threshold)
    reserve_scales = _scales[reserve_index]
    reserve_scores = scores[reserve_index]
    reserve_boxes = boxes[reserve_index]
    logger.info(f"reserve_scales:\n {reserve_scales}")
    logger.info(f"reserve_scores:\n {reserve_scores}")
    logger.info(f"reserve_boxes:\n {reserve_boxes}")

    if len(reserve_boxes) < target_number:
        logger.warning(f"couldn't find {target_number} boxes, only get {len(reserve_boxes)} boxes.")

    # 获取前 target_number 个匹配结果
    if target_number > 0:
        reserve_boxes = reserve_boxes[:target_number]
    logger.info(f"match boxes:\n {reserve_boxes}")
    return reserve_boxes


if __name__ == "__main__":
    test_box_iou()
    test_sort_boxes()
    test_sort_boxes_center()