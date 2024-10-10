from loguru import logger
from algorithm import pixel_num2object_size
from config import MainConfig, CameraConfig, RingsLocationConfig


def transform_result(
    standard_results: dict,
    cycle_results: dict,
    reference_target_id2offset: dict[int, tuple[float, float]] | None,
):
    defalut_error_distance: float = MainConfig.getattr("defalut_error_distance")

    x_move_threshold = RingsLocationConfig.getattr(
        "x_move_threshold"
    )
    y_move_threshold = RingsLocationConfig.getattr(
        "y_move_threshold"
    )
    standard_result_centers = {
        k: result["center"]
        for k, result in standard_results.items()
    }
    standard_result_offsets = {
        k: result["offset"]
        for k, result in standard_results.items()
    }
    standard_result_distance = {
        k: result["distance"]
        for k, result in standard_results.items()
    }
    cycle_centers = {
        k: result["center"]
        for k, result in cycle_results.items()
    }
    logger.info(
        f"standard_result_centers: {standard_result_centers}"
    )
    logger.info(
        f"standard_result_offsets: {standard_result_offsets}"
    )
    logger.info(
        f"standard_result_distance: {standard_result_distance}"
    )
    logger.info(
        f"cycle_centers: {cycle_centers}"
    )

    # 计算移动距离
    distance_result = {}
    for res_k in standard_results.keys():
        if (
            res_k in cycle_centers.keys()
            and cycle_centers[res_k] is not None
            and standard_result_distance[res_k]
            is not None
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
            logger.error(
                f"box {res_k} not found in cycle_centers."
            )

    # TODO: 参考靶标进行滤波处理
    if reference_target_id2offset is not None:
        ref_id: int = int(
            list(reference_target_id2offset.keys())[0]
        )
        if ref_id in distance_result.keys():
            # 找到参考靶标
            ref_distance_x, ref_distance_y = (
                distance_result[ref_id]
            )
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
                RingsLocationConfig.setattr(
                    "reference_target_id2offset",
                    {ref_id: [ref_distance_x, ref_distance_y]},
                )
                logger.info(
                    f"use reference box {ref_id} to calibrate other targets."
                )
                for idx, (
                    distance_x,
                    distance_y,
                ) in distance_result.items():
                    if idx != ref_id:
                        new_distance_x = (
                            distance_x - ref_distance_x
                        )
                        new_distance_y = (
                            distance_y - ref_distance_y
                        )
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
        logger.warning(
            "no reference box set, can't calibrate other targets."
        )

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

    return distance_result, over_distance_ids
