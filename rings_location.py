import numpy as np
from loguru import logger
from pathlib import Path
from algorithm import (
    adaptive_threshold_rings_location,
    pixel_num2object_distance,
)
from config import (
    MainConfig,
    MatchTemplateConfig,
    RingsLocationConfig,
    CameraConfig,
)
from utils import save_to_jsonl


def rings_location(
    image: np.ndarray,
    box_id: int,
    camera_boxestate: list | None,
    image_timestamp: str,
    image_metadata: dict,
):
    location_save_dir: Path = MainConfig.getattr("location_save_dir")
    camera_result_save_path: Path = MainConfig.getattr("camera_result_save_path")
    gradient_threshold_percent: float = RingsLocationConfig.getattr(
        "gradient_threshold_percent"
    )
    iters: int = RingsLocationConfig.getattr("iters")
    order: int = RingsLocationConfig.getattr("order")
    rings_nums: int = RingsLocationConfig.getattr("rings_nums")
    min_group_size: int = RingsLocationConfig.getattr("min_group_size")
    sigmas: int = RingsLocationConfig.getattr("sigmas")
    draw_scale: int = RingsLocationConfig.getattr("draw_scale")
    save_grads: bool = RingsLocationConfig.getattr("save_grads")
    save_detect_images: bool = RingsLocationConfig.getattr("save_detect_images")
    save_detect_results: bool = RingsLocationConfig.getattr("save_detect_results")
    exposure_time = image_metadata["ExposureTime"]

    box: list | None = camera_boxestate["box"]
    try:
        # box 可能为 None, 使用 try except 处理
        x1, y1, x2, y2 = box
        target = image[y1:y2, x1:x2]

        logger.info(f"box {box_id} rings location start")
        result = adaptive_threshold_rings_location(
            target,
            f"camera0--image--{image_timestamp}--{box_id}",
            iters,
            order,
            rings_nums,
            min_group_size,
            sigmas,
            location_save_dir,
            draw_scale,
            save_grads,
            save_detect_images,
            save_detect_results,
            gradient_threshold_percent,
        )
        logger.success(f"{result = }")
        result["metadata"] = image_metadata
        # 保存到文件
        save_to_jsonl(result, camera_result_save_path)
        logger.success(f"box {box_id} rings location success")

        center = [
            float(result["center_x_mean"] + box[0]),
            float(result["center_y_mean"] + box[1]),
        ]
        if np.any(np.isnan(center)):
            center = None
            logger.warning(f"box {box_id} center is nan")

        radii: list[float] = [float(radius) for radius in result["radii"]]
        if np.any(np.isnan(radii)):
            radii = None

        # 根据最大圆环直径计算距离
        distance: float = 0
        if radii is not None:
            distance = pixel_num2object_distance(
                radii[-1] * 2,  # 半径转为直径
                CameraConfig.getattr("pixel_size"),
                CameraConfig.getattr("focus"),
                MatchTemplateConfig.getattr("template_circles_size")[0],
            )

        result = {
            "image_timestamp": f"camera0--image--{image_timestamp}--{box_id}",
            "box": box,
            "center": center,
            "radii": radii,
            "distance": distance,
            "exposure_time": exposure_time,
            "offset": [0, 0],
        }
    except Exception as e:
        logger.error(f"box {box_id} rings location failed, error: {e}")
        result = {
            "image_timestamp": f"camera0--image--{image_timestamp}--{box_id}",
            "box": box,
            "center": None,  # 丢失目标, 置为 None
            "radii": None,
            "distance": None,
            "exposure_time": exposure_time,
            "offset": [0, 0],
        }

    finally:
        return result
