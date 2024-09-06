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


def find_target(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    # 匹配参数
    template_path: Path = MatchTemplateConfig.getattr("template_path")
    match_method: int = MatchTemplateConfig.getattr("match_method")
    init_scale: float = MatchTemplateConfig.getattr("init_scale")
    scales: tuple[float] = MatchTemplateConfig.getattr("scales")
    target_number: int = MatchTemplateConfig.getattr("target_number")
    iou_threshold: float = MatchTemplateConfig.getattr("iou_threshold")
    use_threshold_match: bool = MatchTemplateConfig.getattr("use_threshold_match")
    threshold_match_threshold: float = MatchTemplateConfig.getattr("threshold_match_threshold")
    threshold_iou_threshold: float = MatchTemplateConfig.getattr("threshold_iou_threshold")
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
    # 排序 box，不是必须的
    sorted_index = sort_boxes_center(boxes, sort_by='y')
    sorted_ratios = ratios[sorted_index]
    sorted_scores = scores[sorted_index]
    sorted_boxes = boxes[sorted_index]
    got_target_number = len(sorted_boxes)

    MatchTemplateConfig.setattr("ratios", sorted_ratios)
    MatchTemplateConfig.setattr("scores", sorted_scores)
    MatchTemplateConfig.setattr("boxes", sorted_boxes)
    MatchTemplateConfig.setattr("got_target_number", got_target_number)

    return sorted_ratios, sorted_scores, sorted_boxes, got_target_number
