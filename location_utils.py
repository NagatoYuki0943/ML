from typing import Any
import numpy as np
from loguru import logger
from pathlib import Path
from algorithm import (
    adaptive_threshold_rings_location,
    pixel_num2object_distance,
    pixel_num2object_size,
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
) -> dict | dict[str, Any]:
    """圆环定位封装"""
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


def init_standard_results(
    cycle_results: dict,
    standard_results: dict | None,
    reference_target_id2offset: dict[int, tuple[float, float]] | None = None,
) -> dict | None:
    """更新标准结果"""
    cycle_centers = {k: result["center"] for k, result in cycle_results.items()}
    if len(cycle_centers) == 0:
        logger.warning("no center found in cycle_centers, can't init standard_results.")
        return None
    elif any(v is None for v in cycle_centers.values()):
        logger.warning(
            "some center not found in cycle_centers, can't init standard_results."
        )
        return None
    else:
        if standard_results is None:
            # 标准靶标为 None, 则初始化为 cycle_results
            standard_results = cycle_results
            logger.info(f"init standard_results: {standard_results}")
        else:
            # 标准靶标已经初始化，则添加新的靶标
            for n_k in cycle_results.keys():
                if n_k not in standard_results.keys():
                    standard_results[n_k] = cycle_results[n_k]

            # 更新参考靶标的标准位置
            if reference_target_id2offset is not None:
                ref_id: int = list(reference_target_id2offset.keys())[0]
                if ref_id in standard_results.keys() and ref_id in cycle_results.keys():
                    standard_results[ref_id] = cycle_results[ref_id]
                    logger.info(f"update reference target {ref_id}")

            logger.info(f"update standard_results: {standard_results}")

        return standard_results


def calc_move_distance(
    standard_results: dict,
    cycle_results: dict,
    reference_target_id2offset: dict[int, tuple[float, float]] | None = None,
) -> tuple[dict, set, dict[int, tuple[float, float]] | None]:
    """计算移动距离"""
    defalut_error_distance: float = MainConfig.getattr("defalut_error_distance")

    x_move_threshold = RingsLocationConfig.getattr("x_move_threshold")
    y_move_threshold = RingsLocationConfig.getattr("y_move_threshold")
    standard_result_centers = {
        k: result["center"] for k, result in standard_results.items()
    }
    standard_result_offsets = {
        k: result["offset"] for k, result in standard_results.items()
    }
    standard_result_distance = {
        k: result["distance"] for k, result in standard_results.items()
    }
    cycle_centers = {k: result["center"] for k, result in cycle_results.items()}
    logger.info(f"standard_result_centers: {standard_result_centers}")
    logger.info(f"standard_result_offsets: {standard_result_offsets}")
    logger.info(f"standard_result_distance: {standard_result_distance}")
    logger.info(f"cycle_centers: {cycle_centers}")

    # 计算移动距离
    distance_result = {}
    for res_k in standard_results.keys():
        if (
            res_k in cycle_centers.keys()
            and cycle_centers[res_k] is not None
            and standard_result_distance[res_k] is not None
        ):
            # 移动距离 = 当前位置 - 标准位置 - 补偿值
            pixel_distance_x: float = (
                cycle_centers[res_k][0]
                - standard_result_centers[res_k][0]
                - standard_result_offsets[res_k][0]
            )
            real_distance_x: float = pixel_num2object_size(
                pixel_distance_x,
                standard_result_distance[res_k],
                CameraConfig.getattr("pixel_size"),
                CameraConfig.getattr("focus"),
            )
            # y 轴方向相反
            pixel_distance_y: float = -(
                cycle_centers[res_k][1]
                - standard_result_centers[res_k][1]
                - standard_result_offsets[res_k][1]
            )
            real_distance_y: float = pixel_num2object_size(
                pixel_distance_y,
                standard_result_distance[res_k],
                CameraConfig.getattr("pixel_size"),
                CameraConfig.getattr("focus"),
            )
            logger.info(
                f"box {res_k} move {pixel_distance_x = } pixel, {real_distance_x = } mm, distance = {standard_result_distance[res_k]} mm"
            )
            logger.info(
                f"box {res_k} move {pixel_distance_y = } pixel, {real_distance_y = } mm, distance = {standard_result_distance[res_k]} mm"
            )
            distance_result[res_k] = (
                real_distance_x,
                real_distance_y,
            )
        else:
            # box没找到将移动距离设置为 一个很大的数
            distance_result[res_k] = (
                defalut_error_distance,
                defalut_error_distance,
            )
            logger.error(f"box {res_k} not found in cycle_centers.")

    # TODO: 参考靶标进行滤波处理
    if reference_target_id2offset is not None:
        ref_id: int = int(list(reference_target_id2offset.keys())[0])
        if ref_id in distance_result.keys():
            # 找到参考靶标
            ref_distance_x, ref_distance_y = distance_result[ref_id]
            if (
                abs(ref_distance_x) >= defalut_error_distance
                or abs(ref_distance_y) >= defalut_error_distance
            ):
                # 参考靶标出错
                logger.warning(
                    f"reference box {ref_id} detect failed, can't calibrate other targets."
                )
            else:
                # 参考靶标正常
                # 更新参考靶标的偏移值
                reference_target_id2offset = {ref_id: [ref_distance_x, ref_distance_y]}
                logger.info(f"use reference box {ref_id} to calibrate other targets.")
                for idx, (
                    distance_x,
                    distance_y,
                ) in distance_result.items():
                    if idx != ref_id:
                        new_distance_x = distance_x - ref_distance_x
                        new_distance_y = distance_y - ref_distance_y
                        distance_result[idx] = (
                            new_distance_x,
                            new_distance_y,
                        )
                        logger.info(
                            f"box {idx} after reference, move {new_distance_x = } mm, {new_distance_y = } mm"
                        )
        else:
            logger.warning(
                f"reference box {ref_id} not found in distance_result, can't calibrate other targets."
            )
    else:
        logger.warning("no reference box set, can't calibrate other targets.")

    # 超出距离的 box idx
    over_distance_ids = set()
    for idx, (
        distance_x,
        distance_y,
    ) in distance_result.items():
        if abs(distance_x) > x_move_threshold:
            over_distance_ids.add(idx)
            logger.warning(
                f"box {idx} x move distance {distance_x} mm is over threshold {x_move_threshold} mm."
            )
        else:
            logger.info(
                f"box {idx} x move distance {distance_x} mm is under threshold {x_move_threshold} mm."
            )

        if abs(distance_y) > y_move_threshold:
            over_distance_ids.add(idx)
            logger.warning(
                f"box {idx} y move distance {distance_y} mm is over threshold {y_move_threshold} mm."
            )
        else:
            logger.info(
                f"box {idx} y move distance {distance_y} mm is under threshold {y_move_threshold} mm."
            )

    return distance_result, over_distance_ids, reference_target_id2offset
