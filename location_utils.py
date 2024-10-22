from copy import deepcopy
from typing import Any
import numpy as np
from loguru import logger
from pathlib import Path
from algorithm import (
    adaptive_threshold_rings_location,
    pixel_num2object_distance,
    pixel_num2object_size,
    DualStereoCalibration,
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
        rings_location_result = adaptive_threshold_rings_location(
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
        logger.success(f"{rings_location_result = }")
        rings_location_result["metadata"] = image_metadata
        # 保存到文件
        save_to_jsonl(rings_location_result, camera_result_save_path)
        logger.success(f"box {box_id} rings location success")

        center = [
            float(rings_location_result["center_x_mean"] + box[0]),
            float(rings_location_result["center_y_mean"] + box[1]),
        ]
        if np.any(np.isnan(center)):
            center = None
            logger.warning(f"box {box_id} center is nan")

        radii: list[float] = [
            float(radius) for radius in rings_location_result["radii"]
        ]
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
) -> tuple[dict, dict, dict, dict, set, dict[int, tuple[float, float]] | None]:
    """计算移动距离"""
    defalut_error_distance: float = MainConfig.getattr("defalut_error_distance")

    x_move_threshold: float = RingsLocationConfig.getattr("x_move_threshold")
    y_move_threshold: float = RingsLocationConfig.getattr("y_move_threshold")
    ndigits: int = RingsLocationConfig.getattr("ndigits")

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
    pixel_move_result = {}
    real_move_result = {}
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
            real_distance_x = round(real_distance_x, ndigits)
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
            real_distance_y = round(real_distance_y, ndigits)
            logger.info(
                f"box {res_k} move {pixel_distance_x = } pixel, {real_distance_x = } mm, distance = {standard_result_distance[res_k]} mm"
            )
            logger.info(
                f"box {res_k} move {pixel_distance_y = } pixel, {real_distance_y = } mm, distance = {standard_result_distance[res_k]} mm"
            )
            pixel_move_result[res_k] = (
                pixel_distance_x,
                pixel_distance_y,
            )
            real_move_result[res_k] = (
                real_distance_x,
                real_distance_y,
            )
        else:
            # box没找到将移动距离设置为 一个很大的数
            pixel_move_result[res_k] = (
                defalut_error_distance,
                defalut_error_distance,
            )
            real_move_result[res_k] = (
                defalut_error_distance,
                defalut_error_distance,
            )
            logger.error(f"box {res_k} not found in cycle_centers.")

    pixel_move_result_without_ref = deepcopy(pixel_move_result)
    real_move_result_without_ref = deepcopy(real_move_result)

    # TODO: 参考靶标进行滤波处理
    if reference_target_id2offset is not None:
        ref_id: int = int(list(reference_target_id2offset.keys())[0])
        if ref_id in real_move_result.keys():
            # 找到参考靶标
            ref_pixel_distance_x, ref_pixel_distance_y = pixel_move_result[ref_id]
            ref_real_distance_x, ref_real_distance_y = real_move_result[ref_id]
            if (
                abs(ref_real_distance_x) >= defalut_error_distance
                or abs(ref_real_distance_y) >= defalut_error_distance
            ):
                # 参考靶标出错
                logger.warning(
                    f"reference box {ref_id} detect failed, can't calibrate other targets."
                )
            else:
                # 参考靶标正常
                # 更新参考靶标的偏移值
                reference_target_id2offset = {
                    ref_id: [ref_real_distance_x, ref_real_distance_y]
                }
                logger.info(f"use reference box {ref_id} to calibrate other targets.")

                # 计算像素偏移
                for idx, (
                    pixel_distance_x,
                    pixel_distance_y,
                ) in pixel_move_result.items():
                    if idx != ref_id:
                        new_pixel_distance_x: float = (
                            pixel_distance_x - ref_pixel_distance_x
                        )
                        new_pixel_distance_y: float = (
                            pixel_distance_y - ref_pixel_distance_y
                        )
                        pixel_move_result[idx] = (
                            new_pixel_distance_x,
                            new_pixel_distance_y,
                        )
                        logger.info(
                            f"box {idx} after reference, move {new_pixel_distance_x = } pixel, {new_pixel_distance_y = } pixel"
                        )

                # 计算真实偏移
                for idx, (
                    real_distance_x,
                    real_distance_y,
                ) in real_move_result.items():
                    if idx != ref_id:
                        new_real_distance_x: float = round(
                            real_distance_x - ref_real_distance_x, ndigits
                        )
                        new_real_distance_y: float = round(
                            real_distance_y - ref_real_distance_y, ndigits
                        )
                        real_move_result[idx] = (
                            new_real_distance_x,
                            new_real_distance_y,
                        )
                        logger.info(
                            f"box {idx} after reference, move {new_real_distance_x = } mm, {new_real_distance_y = } mm"
                        )
        else:
            logger.warning(
                f"reference box {ref_id} not found in real_move_result, can't calibrate other targets."
            )
    else:
        logger.warning("no reference box set, can't calibrate other targets.")

    # 超出距离的 box idx
    over_threshold_ids = set()
    for idx, (
        real_distance_x,
        real_distance_y,
    ) in real_move_result.items():
        if abs(real_distance_x) > x_move_threshold:
            over_threshold_ids.add(idx)
            logger.warning(
                f"box {idx} x move distance {real_distance_x} mm is over threshold {x_move_threshold} mm."
            )
        else:
            logger.info(
                f"box {idx} x move distance {real_distance_x} mm is under threshold {x_move_threshold} mm."
            )

        if abs(real_distance_y) > y_move_threshold:
            over_threshold_ids.add(idx)
            logger.warning(
                f"box {idx} y move distance {real_distance_y} mm is over threshold {y_move_threshold} mm."
            )
        else:
            logger.info(
                f"box {idx} y move distance {real_distance_y} mm is under threshold {y_move_threshold} mm."
            )

    return (
        pixel_move_result,
        pixel_move_result_without_ref,
        real_move_result,
        real_move_result_without_ref,
        over_threshold_ids,
        reference_target_id2offset,
    )


def calc_z_distance(
    left_camera_result: dict,
    right_camera_result: dict,
    dual_stereo_calibration: DualStereoCalibration,
) -> dict:
    left_camera_centers = {
        k: result["center"] for k, result in left_camera_result.items()
    }
    right_camera_centers = {
        k: result["center"] for k, result in right_camera_result.items()
    }

    z_distance = {}
    for k in left_camera_centers.keys():
        if (
            k in right_camera_centers.keys()
            and left_camera_centers[k] is not None
            and right_camera_centers[k] is not None
        ):
            left_center: list[float] = left_camera_centers[k]
            right_center: list[float] = right_camera_centers[k]
            avg_disparity, avg_depth, depths, focal_length_mm = (
                dual_stereo_calibration.pixel_to_world(left_center, right_center)
            )
            z_distance[k] = avg_depth

        else:
            z_distance[k] = 0

    return z_distance


def compare_z_distance(
    z_distance1: dict,
    z_distance2: dict,
) -> tuple[dict, set]:
    defalut_error_distance: float = MainConfig.getattr("defalut_error_distance")
    z_move_threshold: float = RingsLocationConfig.getattr("z_move_threshold")
    z_move_distance = {}
    ndigits: int = RingsLocationConfig.getattr("ndigits")

    over_threshold_ids = set()
    for k in z_distance1.keys():
        if k in z_distance2.keys():
            move: float = round(z_distance1[k] - z_distance2[k], ndigits)
            z_move_distance[k] = move
            if abs(move) > z_move_threshold:
                over_threshold_ids.add(k)
                logger.warning(
                    f"box {k} z move distance {move} mm is over threshold {z_move_threshold} mm."
                )
            else:
                logger.info(
                    f"box {k} z move distance {move} mm is under threshold {z_move_threshold} mm."
                )
        else:
            z_move_distance[k] = defalut_error_distance
            logger.warning(f"box {k} not found in z_distance2.")

    return z_move_distance, over_threshold_ids
