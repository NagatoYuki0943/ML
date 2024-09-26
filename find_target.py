import numpy as np
from pathlib import Path
from loguru import logger
import cv2

from algorithm import (
    match_template_filter_by_threshold,
    multi_target_multi_scale_match_template,
    sort_boxes_center,
)
from config import MatchTemplateConfig


# def find_target(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
#     logger.info("find target start")
#     # 匹配参数
#     template_path: Path = MatchTemplateConfig.getattr("template_path")
#     match_method: int = MatchTemplateConfig.getattr("match_method")
#     init_scale: float = MatchTemplateConfig.getattr("init_scale")
#     scales: tuple[float] = MatchTemplateConfig.getattr("scales")
#     target_number: int = MatchTemplateConfig.getattr("target_number")
#     iou_threshold: float = MatchTemplateConfig.getattr("iou_threshold")
#     use_threshold_match: bool = MatchTemplateConfig.getattr("use_threshold_match")
#     threshold_match_threshold: float = MatchTemplateConfig.getattr("threshold_match_threshold")
#     threshold_iou_threshold: float = MatchTemplateConfig.getattr("threshold_iou_threshold")
#     template = cv2.imread(template_path, 0)

#     # ratios: [...]
#     # scores: [...]
#     # boxes: [[x_min, y_min, x_max, y_max], ...]
#     ratios, scores, boxes = multi_target_multi_scale_match_template(
#         image,
#         template,
#         match_method,
#         init_scale,
#         scales,
#         target_number,
#         iou_threshold,
#         use_threshold_match,
#         threshold_match_threshold,
#         threshold_iou_threshold,
#     )
#     # 排序 box，不是必须的
#     sorted_index = sort_boxes_center(boxes, sort_by='y')
#     sorted_ratios = ratios[sorted_index]
#     sorted_scores = scores[sorted_index]
#     sorted_boxes = boxes[sorted_index]
#     got_target_number = len(sorted_boxes)

#     MatchTemplateConfig.setattr("ratios", sorted_ratios)
#     MatchTemplateConfig.setattr("scores", sorted_scores)
#     MatchTemplateConfig.setattr("boxes", sorted_boxes)
#     MatchTemplateConfig.setattr("got_target_number", got_target_number)

#     if got_target_number < target_number:
#         logger.error(f"find target number less than target number, got_target_number: {got_target_number}, target_number: {target_number}")
#     else:
#         logger.success(f"find target number {got_target_number} = set target number {target_number}")

#     logger.info("find target end")
#     return sorted_ratios, sorted_scores, sorted_boxes, got_target_number


# # 在原来box周围扩大范围搜寻
# def around_find_target(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
#     logger.info("around find target start")
#     # 匹配参数
#     template_path: Path = MatchTemplateConfig.getattr("template_path")
#     match_method: int = MatchTemplateConfig.getattr("match_method")
#     target_number: int = MatchTemplateConfig.getattr("target_number")
#     search_range: float = MatchTemplateConfig.getattr("search_range")
#     template = cv2.imread(template_path, 0)

#     # 获取之前的匹配结果
#     ratios: np.ndarray = MatchTemplateConfig.getattr("ratios")
#     boxes: np.ndarray = MatchTemplateConfig.getattr("boxes")

#     image_h, image_w = image.shape[:2]
#     template_h, template_w = template.shape[:2]

#     new_ratios = []
#     new_scores = []
#     new_boxes = []

#     # 循环box，截取box区域进行匹配
#     for i, (ratio, box) in enumerate(zip(ratios, boxes)):
#         # 匹配区域
#         box_x1, box_y1, box_x2, box_y2 = box
#         box_target = image[box_y1:box_y2, box_x1:box_x2]

#         # 根据最终比率得到模板的尺寸
#         resized_h = int(ratio * template_h)
#         resized_w = int(ratio * template_w)
#         # 使用最终尺寸一次 resize 模板
#         template_resized = cv2.resize(template, (resized_w, resized_h))

#         # 需要调整模板尺度
#         match_results = match_template_filter_by_threshold(
#             box_target,
#             template_resized,
#             match_method,
#         )

#         if len(match_results) > 0:
#             # 找到目标
#             new_score, new_box = match_results[0]
#             new_x1, new_y1, new_x2, new_y2 = new_box
#             new_ratios.append(ratio)
#             new_scores.append(new_score)
#             new_boxes.append([box_x1 + new_x1, box_y1 + new_y1, box_x1 + new_x2, box_y1 + new_y2])
#             logger.info(f"original target {i}, {ratio = }, {box = } is ok")
#         else:
#             # 没有找到目标,扩大匹配区域
#             box_h = box_y2 - box_y1
#             box_w = box_x2 - box_x1
#             box_x1_dilate = max(int(box_x1 - search_range * box_w), 0)
#             box_x2_dilate = min(int(box_x2 + search_range * box_w), image_w)
#             box_y1_dilate = max(int(box_y1 - search_range * box_h), 0)
#             box_y2_dilate = min(int(box_y2 + search_range * box_h), image_h)
#             box_target_dilate = image[box_y1_dilate:box_y2_dilate, box_x1_dilate:box_x2_dilate]
#             # 需要调整模板尺度
#             match_results = match_template_filter_by_threshold(
#                 box_target_dilate,
#                 template_resized,
#                 match_method,
#             )
#             if len(match_results) > 0:
#                 # 找到目标
#                 new_score, new_box = match_results[0]
#                 new_x1, new_y1, new_x2, new_y2 = new_box
#                 new_ratios.append(ratio)
#                 new_scores.append(new_score)
#                 new_boxes.append([box_x1 + new_x1, box_y1 + new_y1, box_x1 + new_x2, box_y1 + new_y2])
#                 logger.info(f"original target {i}, {ratio = }, {box = } not found, but found in box around, {new_box = }")
#             else:
#                 # 没有找到目标
#                 logger.warning(f"original target {i}, {ratio = }, {box = } not found")

#     new_ratios = np.array(new_ratios)
#     new_scores = np.array(new_scores)
#     new_boxes = np.array(new_boxes)

#     # 排序 box，不是必须的
#     sorted_index = sort_boxes_center(boxes, sort_by='y')
#     sorted_ratios = new_ratios[sorted_index]
#     sorted_scores = new_scores[sorted_index]
#     sorted_boxes = new_boxes[sorted_index]
#     got_target_number = len(sorted_boxes)

#     MatchTemplateConfig.setattr("ratios", sorted_ratios)
#     MatchTemplateConfig.setattr("scores", sorted_scores)
#     MatchTemplateConfig.setattr("boxes", sorted_boxes)
#     MatchTemplateConfig.setattr("got_target_number", got_target_number)

#     if got_target_number < target_number:
#         logger.error(f"find target number less than target number, got_target_number: {got_target_number}, target_number: {target_number}")
#     else:
#         logger.success(f"find target number {got_target_number} = set target number {target_number}")

#     logger.info("around find target end")
#     return sorted_ratios, sorted_scores, sorted_boxes, got_target_number


def find_target(image: np.ndarray, camera_index: int = 0) -> tuple[dict, int]:
    logger.info("find target start")
    # 匹配参数
    template_path: Path = MatchTemplateConfig.getattr("template_path")
    match_method: int = MatchTemplateConfig.getattr("match_method")
    init_scale: float = MatchTemplateConfig.getattr("init_scale")
    scales: tuple[float] = MatchTemplateConfig.getattr("scales")
    target_number: int = MatchTemplateConfig.getattr("target_number")
    iou_threshold: float = MatchTemplateConfig.getattr("iou_threshold")
    use_threshold_match: bool = MatchTemplateConfig.getattr("use_threshold_match")
    threshold_match_threshold: float = MatchTemplateConfig.getattr(
        "threshold_match_threshold"
    )
    threshold_iou_threshold: float = MatchTemplateConfig.getattr(
        "threshold_iou_threshold"
    )
    template = cv2.imread(template_path, 0)

    # ratios: [...]
    # scores: [...]
    # boxes: [[x_min, y_min, x_max, y_max], ...]
    ratios, scores, boxes = multi_target_multi_scale_match_template(
        image,
        template,
        match_method,
        init_scale,
        scales,
        target_number,
        iou_threshold,
        use_threshold_match,
        threshold_match_threshold,
        threshold_iou_threshold,
    )
    # 没有找到任何目标
    if len(ratios) == 0:
        logger.warning("can not find any target")
        MatchTemplateConfig.setattr("id2boxstate", None)
        MatchTemplateConfig.setattr("got_target_number", 0)
        return None, 0

    # 排序 box，不是必须的
    sorted_index = sort_boxes_center(boxes, sort_by="y")
    sorted_ratios = ratios[sorted_index]
    sorted_scores = scores[sorted_index]
    sorted_boxes = boxes[sorted_index]

    # 转换为字典保存
    id2boxstate = {}
    for i, (ratio, score, box) in enumerate(
        zip(sorted_ratios, sorted_scores, sorted_boxes)
    ):
        id2boxstate[i] = {
            "ratio": float(ratio),
            "score": float(score),
            "box": box.tolist(),
        }

    MatchTemplateConfig.setattr("id2boxstate", id2boxstate)
    got_target_number = len(id2boxstate)
    MatchTemplateConfig.setattr("got_target_number", got_target_number)

    if got_target_number < target_number:
        logger.warning(
            f"find target number: {got_target_number} less than target number: {target_number}"
        )
    elif got_target_number > target_number:
        logger.warning(
            f"find target number: {got_target_number} more than target number: {got_target_number}"
        )
        if target_number == 0:
            logger.warning(
                f"target_number is 0, use got_target_number: {got_target_number} as target_number"
            )
            MatchTemplateConfig.setattr("target_number", got_target_number)
    else:
        logger.success(
            f"find target number {got_target_number} = set target number {target_number}"
        )

    logger.info("find target end")
    return id2boxstate, got_target_number


# 在原来box周围扩大范围搜寻
def find_around_target(image: np.ndarray, camera_index: int = 0) -> tuple[dict, int]:
    logger.info("find around target start")
    # 匹配参数
    template_path: Path = MatchTemplateConfig.getattr("template_path")
    match_method: int = MatchTemplateConfig.getattr("match_method")
    target_number: int = MatchTemplateConfig.getattr("target_number")
    new_target_scales: tuple[float] = MatchTemplateConfig.getattr("new_target_scales")
    threshold_match_threshold: float = MatchTemplateConfig.getattr(
        "threshold_match_threshold"
    )
    search_range: float = MatchTemplateConfig.getattr("search_range")
    template = cv2.imread(template_path, 0)

    # 获取之前的匹配结果
    # id2boxstate: {
    #     i: {
    #         "ratio": ratio,
    #         "score": score,
    #         "box": box
    #     }
    # }
    id2boxstate: dict | None = MatchTemplateConfig.getattr("id2boxstate")
    # 如果没有目标，则直接全图查找
    if id2boxstate is None:
        logger.warning("id2boxstate is None, use find_target")
        return find_target(image)

    image_h, image_w = image.shape[:2]
    template_h, template_w = template.shape[:2]

    new_id2boxstate = {}

    # 循环box，截取box区域进行匹配
    for i, boxestate in id2boxstate.items():
        ratio: float | None = boxestate["ratio"]
        score: float | None = boxestate["score"]
        box: list[int] | None = boxestate["box"]

        # box 为 None 则跳过
        if box is None:
            new_id2boxstate[i] = {"ratio": ratio, "score": score, "box": None}
            continue

        # 获取 box 坐标
        box_x1, box_y1, box_x2, box_y2 = box

        # ratio 为 None 代表这个 box 是强制指定的，需要计算 ratio
        if ratio is None:
            box_h = box_y2 - box_y1
            box_w = box_x2 - box_x1
            # 求出 ratio 比例
            # ratio = min(box_h, box_w) / min(template_h, template_w)
            ratio = min(box_h / template_h, box_w / template_w)
            new_scales = np.arange(
                new_target_scales[0], new_target_scales[1] + 1e-8, new_target_scales[2]
            )
            ratios = (ratio * new_scales).tolist()
            logger.info(
                f"find around target {i = } original ratio is None, use {ratios = } to search"
            )
        else:
            ratios = [ratio]

        # 匹配区域
        box_target = image[box_y1:box_y2, box_x1:box_x2]

        # 多个 ratio 尝试匹配
        match_results = []
        for _ratio in ratios:
            # 根据最终比率得到模板的尺寸
            resized_h = int(_ratio * template_h)
            resized_w = int(_ratio * template_w)
            # 使用最终尺寸一次 resize 模板
            template_resized = cv2.resize(template, (resized_w, resized_h))

            # 需要调整模板尺度
            match_result = match_template_filter_by_threshold(
                box_target,
                template_resized,
                match_method,
                threshold_match_threshold,
            )
            match_result = [
                [_ratio] + list(_match_result) for _match_result in match_result
            ]
            match_results.extend(match_result)

        # 按照 score 降序排序 match_results
        match_results = sorted(match_results, key=lambda x: x[1], reverse=True)

        if len(match_results) > 0:
            # 找到目标
            new_ratio, new_score, new_box = match_results[0]
            new_x1, new_y1, new_x2, new_y2 = new_box.tolist()
            new_box = [
                box_x1 + new_x1,
                box_y1 + new_y1,
                box_x1 + new_x2,
                box_y1 + new_y2,
            ]
            new_id2boxstate[i] = {
                "ratio": float(new_ratio),
                "score": float(new_score),
                "box": new_box,
            }
            logger.info(
                f"original target {i}, {new_ratio = }, {new_score = }, {new_box = } is ok"
            )
        else:
            logger.warning(
                f"original target {i}, {ratio = }, {score = }, {box = } not found, dilate search range"
            )
            # 没有找到目标,扩大匹配区域
            box_h = box_y2 - box_y1
            box_w = box_x2 - box_x1
            box_x1_dilate = max(int(box_x1 - search_range * box_w), 0)
            box_x2_dilate = min(int(box_x2 + search_range * box_w), image_w)
            box_y1_dilate = max(int(box_y1 - search_range * box_h), 0)
            box_y2_dilate = min(int(box_y2 + search_range * box_h), image_h)
            box_target_dilate = image[
                box_y1_dilate:box_y2_dilate, box_x1_dilate:box_x2_dilate
            ]

            # 多个 ratio 尝试匹配
            match_results = []
            for _ratio in ratios:
                # 根据最终比率得到模板的尺寸
                resized_h = int(_ratio * template_h)
                resized_w = int(_ratio * template_w)
                # 使用最终尺寸一次 resize 模板
                template_resized = cv2.resize(template, (resized_w, resized_h))

                # 需要调整模板尺度
                match_result = match_template_filter_by_threshold(
                    box_target_dilate,
                    template_resized,
                    match_method,
                    threshold_match_threshold,
                )
                match_result = [
                    [_ratio] + list(_match_result) for _match_result in match_result
                ]
                match_results.extend(match_result)

            # 按照 score 降序排序 match_results
            match_results = sorted(match_results, key=lambda x: x[1], reverse=True)

            if len(match_results) > 0:
                # 找到目标
                new_ratio, new_score, new_box = match_results[0]
                new_x1, new_y1, new_x2, new_y2 = new_box.tolist()
                new_box = [
                    box_x1 + new_x1,
                    box_y1 + new_y1,
                    box_x1 + new_x2,
                    box_y1 + new_y2,
                ]
                new_id2boxstate[i] = {
                    "ratio": float(new_ratio),
                    "score": float(new_score),
                    "box": new_box,
                }
                logger.info(
                    f"original target {i}, {ratio = }, {score = }, {box = } not found, but found in dilate range, {new_ratio = }, {new_box = }"
                )
            else:
                # 没有找到目标
                new_id2boxstate[i] = {
                    "ratio": float(ratio),
                    "score": float(score),
                    "box": None,
                }
                logger.warning(
                    f"in dilate range, original target {i}, {ratio = }, {score = }, {box = } not found"
                )

    # 如果检测不到全部的 box, 则不设置全局变量
    if all(
        (True if boxestate["box"] is None else False)
        for boxestate in new_id2boxstate.values()
    ):
        logger.warning("find around target failed, no target found")

    MatchTemplateConfig.setattr("id2boxstate", new_id2boxstate)
    # 更新got_target_number
    got_target_number = len(
        [
            boxestate
            for boxestate in new_id2boxstate.values()
            if boxestate["box"] is not None
        ]
    )
    MatchTemplateConfig.setattr("got_target_number", got_target_number)

    if got_target_number < target_number:
        logger.error(
            f"find target number less than target number, got_target_number: {got_target_number}, target_number: {target_number}"
        )
    elif got_target_number > target_number:
        logger.warning(
            f"find target number more than target number, got_target_number: {got_target_number}, target_number: {target_number}, please update config"
        )
    else:
        logger.success(
            f"find target number {got_target_number} = set target number {target_number}"
        )

    logger.info("find around target end")

    return new_id2boxstate, got_target_number


# 全局查找丢失的box，屏蔽已知的box
def find_lost_target(image: np.ndarray, camera_index: int = 0) -> tuple[dict, int]:
    logger.info("find lost target start")
    # 匹配参数
    template_path: Path = MatchTemplateConfig.getattr("template_path")
    match_method: int = MatchTemplateConfig.getattr("match_method")
    init_scale: float = MatchTemplateConfig.getattr("init_scale")
    scales: tuple[float] = MatchTemplateConfig.getattr("scales")
    # target_number: int = MatchTemplateConfig.getattr("target_number")
    iou_threshold: float = MatchTemplateConfig.getattr("iou_threshold")
    use_threshold_match: bool = MatchTemplateConfig.getattr("use_threshold_match")
    threshold_match_threshold: float = MatchTemplateConfig.getattr(
        "threshold_match_threshold"
    )
    threshold_iou_threshold: float = MatchTemplateConfig.getattr(
        "threshold_iou_threshold"
    )
    template = cv2.imread(template_path, 0)

    # 获取之前的匹配结果
    # id2boxstate: {
    #     i: {
    #         "ratio": ratio,
    #         "score": score,
    #         "box": box
    #     }
    # }
    id2boxstate: dict | None = MatchTemplateConfig.getattr("id2boxstate")
    # 如果没有目标，则直接全图查找
    if id2boxstate is None:
        logger.warning("id2boxstate is None, use find_target")
        return find_target(image)

    image = image.copy()
    loss_ids = []
    # 循环box，将box区域屏蔽
    for i, boxestate in id2boxstate.items():
        box: list[int] | None = boxestate["box"]
        if box is None:
            loss_ids.append(i)
            continue
        # 屏蔽其他box
        box_x1, box_y1, box_x2, box_y2 = box
        image[box_y1:box_y2, box_x1:box_x2] = 0
    logger.warning(f"find lost target, loss_ids: {loss_ids}")

    # 查找丢失的目标
    loss_target_number = len(loss_ids)
    # ratios: [...]
    # scores: [...]
    # boxes: [[x_min, y_min, x_max, y_max], ...]
    ratios, scores, boxes = multi_target_multi_scale_match_template(
        image,
        template,
        match_method,
        init_scale,
        scales,
        loss_target_number,
        iou_threshold,
        use_threshold_match,
        threshold_match_threshold,
        threshold_iou_threshold,
    )
    # 如果没有检测到任何丢失的目标，就不会修改原始值
    if len(ratios) == 0:
        logger.warning("find lost target failed, no target found")
        got_target_number = len(
            [
                boxestate
                for boxestate in id2boxstate.values()
                if boxestate["box"] is not None
            ]
        )
        return id2boxstate, got_target_number

    # 排序 box，不是必须的
    sorted_index = sort_boxes_center(boxes, sort_by="y")
    sorted_ratios = ratios[sorted_index]
    sorted_scores = scores[sorted_index]
    sorted_boxes = boxes[sorted_index]

    if len(sorted_ratios) < len(loss_ids):
        logger.error(
            f"find lost target number less than loss target number, find_lost_target_number: {len(sorted_ratios)}, loss_target_number: {len(loss_ids)}"
        )
    else:
        logger.success(
            f"find_lost_target_number {len(sorted_ratios)} = loss_target_number {len(loss_ids)}"
        )

    for i, (ratio, score, box) in enumerate(
        zip(sorted_ratios, sorted_scores, sorted_boxes)
    ):
        id2boxstate[loss_ids[i]] = {
            "ratio": float(ratio),
            "score": float(score),
            "box": box.tolist(),
        }
        logger.info(f"find lost target {loss_ids[i]}, {ratio = }, {score = }, {box = }")

    MatchTemplateConfig.setattr("id2boxstate", id2boxstate)
    # 更新got_target_number
    got_target_number = len(
        [
            boxestate
            for boxestate in id2boxstate.values()
            if boxestate["box"] is not None
        ]
    )
    MatchTemplateConfig.setattr("got_target_number", got_target_number)

    logger.info("find lost target end")

    return id2boxstate, got_target_number
