import numpy as np
from pathlib import Path
from loguru import logger
import cv2

from algorithm import (
    multi_target_multi_scale_match_template,
    sort_boxes_center,
)
from config import MatchTemplateConfig


def find_target(image: np.ndarray)-> np.ndarray:
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

    # [[x_min, y_min, x_max, y_max],...]
    boxes = multi_target_multi_scale_match_template(
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
    sorted_boxes = sort_boxes_center(boxes, sort_by='y')
    return sorted_boxes
