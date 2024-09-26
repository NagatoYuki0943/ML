import os
from pathlib import Path
import time
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any
import pickle
from loguru import logger

from subpixel_edges import subpixel_edges
from .fit_circle_by_least_square import fit_circle_by_least_square_filter
from .split_rings import split_rings_adaptive
from .image_metrics import image_gradient, get_gradient_threshold


def rings_location(
    image: np.ndarray,
    timestamp: Any | None = None,
    subpixel_edges_threshold: int = 25,
    iters: int = 0,
    order: int = 2,
    rings_nums: int = 6,
    min_group_size: int = 3,
    sigmas: list[float | int] | float | int = 0,
    save_dir: str | Path = "./save_dir",
    draw_scale: int = 20,
    save_grads: bool = False,
    save_detect_images: bool = True,
    save_detect_results: bool = True,
) -> dict:
    """定位圆环

    Args:
        image (np.ndarray): 图片
        timestamp: Any | None: 时间戳. Defaults to None.
        subpixel_edges_threshold (int, optional): 子像素梯度阈值. Defaults to 25.
        iters (int, optional): 子像素检测迭代次数，模糊图像可以增加迭代次数. Defaults to 0.
        order (int, optional): 边缘检测使用公式的阶数. Defaults to 2.
        rings_nums (int, optional): 要检测的圆环的数量. Defaults to 6.
        min_group_size (int, optional): 每个圆环最小的点数,小于这个值的圆的组会被丢弃. Defaults to 3.
        sigmas (list[float | int] | float | int, optional): 拟合圆时通过拟合的圆心到每个点的距离进行过滤，是通过误差的正态分布过滤的，<=0 代表禁用过滤. Defaults to 0.
            这个值的含义是 n 个 sigma 的值，有几个值，就过滤几次，每个值代表本次过滤的 sigma 的值
            - 横轴区间（μ-σ,μ+σ）  内的面积为68.268949%
            - 横轴区间（μ-2σ,μ+2σ）内的面积为95.449974%
            - 横轴区间（μ-3σ,μ+3σ）内的面积为99.730020%
        save_dir (str, optional): 保存图片路径. Defaults to "./".
        draw_scale (int, optional): 缩放图片的倍率. Defaults to 20.
        save_grads (bool, optional): 是否保存梯度图. Defaults to False.
        save_detect_images (bool, optional): 是否保存文件. Defaults to True.
        save_detect_results (bool, optional): 是否保存检测结果. Defaults to True.

    Returns:
        dict: 检测结果
    """
    timestamp = (
        time.strftime("%Y%m%d-%H%M%S", time.localtime())
        if timestamp is None
        else timestamp
    )
    os.makedirs(save_dir, exist_ok=True)

    if len(image.shape) == 3:
        # 转化为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测边缘
    logger.info(f"{subpixel_edges_threshold = }")
    edges, grad, absGxInner, absGyInner = subpixel_edges(
        image,
        subpixel_edges_threshold,
        iters,
        order,
    )

    # 保存结果
    if save_grads:
        with open(os.path.join(save_dir, f"{timestamp}-edges.pkl"), "wb") as f:
            pickle.dump(edges, f)
        np.save(os.path.join(save_dir, f"{timestamp}-grad"), grad)
        np.save(os.path.join(save_dir, f"{timestamp}-absGxInner"), absGxInner)
        np.save(os.path.join(save_dir, f"{timestamp}-absGyInner"), absGyInner)

    if save_detect_images:
        plt.imshow(grad[2:-2, 2:-2])  # 忽略边缘
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{timestamp}-grad.png"))
        plt.close()
        plt.imshow(absGxInner)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{timestamp}-absGxInner.png"))
        plt.close()
        plt.imshow(absGyInner)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{timestamp}-absGyInner.png"))
        plt.close()

    # 边缘像素xy坐标
    pixel_edges_xy: np.ndarray = np.stack([edges.pixel_x, edges.pixel_y], axis=1)
    logger.info(f"pixel_edges: {pixel_edges_xy.shape = }")
    if save_detect_images:
        figure = plt.figure(figsize=(6, 6))
        plt.scatter(edges.pixel_x, edges.pixel_y, marker=".")
        plt.grid()
        plt.title("pixel edges")
        figure.savefig(os.path.join(save_dir, f"{timestamp}-fig-pixel-edges.png"))
        plt.close()

    # 边缘子像素xy坐标
    subpixel_edges_xy: np.ndarray = np.stack([edges.x, edges.y], axis=1)
    logger.info(f"subpixel_edges: {subpixel_edges_xy.shape = }")
    if save_detect_images:
        figure = plt.figure(figsize=(6, 6))
        plt.scatter(edges.x, edges.y, marker=".")
        plt.grid()
        plt.title("subpixel edges")
        figure.savefig(os.path.join(save_dir, f"{timestamp}-fig-subpixel-edges.png"))
        plt.close()

    # 分为单个圆环
    group_rings = split_rings_adaptive(
        subpixel_edges_xy,
        rings_nums=rings_nums,
        min_group_size=min_group_size,
        init_threshold_range=0.5,
    )
    if save_detect_images:
        figure = plt.figure(figsize=(6, 6))
        for group_circle in group_rings:
            plt.scatter(group_circle[:, 0], group_circle[:, 1], marker=".")
        plt.grid()
        plt.title("group rings")
        figure.savefig(os.path.join(save_dir, f"{timestamp}-fig-group-rings.png"))
        plt.close()

    # 拟合每个圆环
    center_xs = []
    center_ys = []
    radii = []
    err_avgs = []
    err_vars = []
    err_stds = []
    fit_rings = []
    ignore_rings = []
    for i, group_circle in enumerate(group_rings):
        # logger.info(f"split ring: {i}, pixel numbers = {group_circle.shape[0]}")
        (
            center_x,
            center_y,
            radius,
            err_avg,
            err_var,
            err_std,
            err_abs,
            radii_err,
            fit_circle,
            ignore_circle,
        ) = fit_circle_by_least_square_filter(group_circle, sigmas)
        center_xs.append(center_x)
        center_ys.append(center_y)
        radii.append(radius)
        err_avgs.append(err_avg)
        err_vars.append(err_var)
        err_stds.append(err_std)
        fit_rings.append(fit_circle)
        ignore_rings.append(ignore_circle)

    if save_detect_images:
        # 绘制过滤前和过滤后的图片
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(45, 8))
        # 子像素边缘
        axes[0].scatter(edges.x, edges.y, marker=".")
        axes[0].grid()
        axes[0].set_title("subpixel edges")
        # 分组子像素边缘
        for group_circle in group_rings:
            axes[1].scatter(group_circle[:, 0], group_circle[:, 1], marker=".")
        axes[1].grid()
        axes[1].set_title("group rings")
        # 拟合的圆
        for circle in fit_rings:
            axes[2].scatter(circle[:, 0], circle[:, 1], marker=".")
        for center_x, center_y in zip(center_xs, center_ys):
            axes[2].scatter(center_x, center_y, marker=".")
        axes[2].grid()
        axes[2].set_title("fit rings")
        # 有忽略的圆才画
        if ignore_rings[0] is not None:
            # 忽略的圆
            for circle in ignore_rings:
                axes[3].scatter(circle[:, 0], circle[:, 1], marker=".")
            axes[3].grid()
            axes[3].set_title("ignore rings")
            # 拟合的圆和忽略的圆
            for fit_circle, ignore_circle in zip(fit_rings, ignore_rings):
                axes[4].scatter(fit_circle[:, 0], fit_circle[:, 1], marker=".")
                axes[4].scatter(ignore_circle[:, 0], ignore_circle[:, 1], marker=".")
            axes[4].grid()
            axes[4].set_title("fit & ignore rings")
        fig.savefig(os.path.join(save_dir, f"{timestamp}-fig-compare.png"))
        plt.close()

        # 创建高分辨率图片，用于绘制亚像素边缘
        draw_image = np.repeat(
            np.repeat(image, repeats=draw_scale, axis=0), repeats=draw_scale, axis=1
        )

        # 绘制边缘
        for i, fit_circle in enumerate(fit_rings):
            for j in range(len(fit_circle)):
                # 边缘坐标 + 0.5 的含义是原本偏移坐标是从每个像素的左上角坐标偏移的，+0.5改为了从像素的中心点偏移
                center = np.round((fit_circle[j] + 0.5) * draw_scale).astype(np.int32)
                draw_image = cv2.circle(
                    img=draw_image,
                    center=center,  # center (x, y)
                    radius=1,  # 半径
                    color=(136, 14, 79),  # color
                    thickness=1,  # 线宽
                )

        # 绘制中心点
        for i, (center_x, center_y) in enumerate(zip(center_xs, center_ys)):
            center = [round(center_x * draw_scale), round(center_y * draw_scale)]
            draw_image = cv2.circle(
                img=draw_image,
                center=center,  # center (x, y)
                radius=1,  # 半径
                color=(136, 14, 79),  # color
                thickness=-1,  # 线宽
            )

        # 保存图片
        cv2.imwrite(os.path.join(save_dir, f"{timestamp}-result.png"), draw_image)

    # 求所有圆环信息的均值
    center_x_mean = np.mean(center_xs)
    center_y_mean = np.mean(center_ys)
    logger.info(f"所有圆的圆心坐标均值信息：{center_x_mean = }, {center_y_mean = }")

    if save_detect_results:
        with open(
            os.path.join(save_dir, f"{timestamp}.json"), mode="w", encoding="utf-8"
        ) as f:
            image_data = {
                "timestamp": timestamp,
                "threshold": subpixel_edges_threshold,
                "pixel_edges_xy": pixel_edges_xy.tolist(),
                "subpixel_edges_xy": subpixel_edges_xy.tolist(),
                "group_rings": [circle.tolist() for circle in group_rings],
                "fit_rings": [circle.tolist() for circle in fit_rings],
                "ignore_rings": [
                    (None if circle is None else circle.tolist())
                    for circle in ignore_rings
                ],
                "center_xs": center_xs,
                "center_ys": center_ys,
                "radii": radii,
                "err_avgs": err_avgs,
                "err_vars": err_vars,
                "err_stds": err_stds,
                "center_x_mean": center_x_mean,
                "center_y_mean": center_y_mean,
            }
            json.dump(image_data, f, indent=4, ensure_ascii=False)

    # 最终结果
    measure_result = {
        "timestamp": timestamp,
        "threshold": subpixel_edges_threshold,
        "center_xs": center_xs,
        "center_ys": center_ys,
        "radii": radii,
        "err_avgs": err_avgs,
        "err_vars": err_vars,
        "err_stds": err_stds,
        "center_x_mean": center_x_mean,
        "center_y_mean": center_y_mean,
    }

    return measure_result


def multi_images_rings_location(
    images: list[np.ndarray],
    roi_names: list[int | str],
    timestamp: Any | None = None,
    subpixel_edges_threshold: int = 25,
    iters: int = 0,
    order: int = 2,
    rings_nums: int = 6,
    min_group_size: int = 3,
    sigmas: list[float | int] | float | int = 0,
    save_dir="./save_dir",
    draw_scale: int = 20,
    save_grads: bool = False,
    save_detect_images: bool = True,
    save_detect_results: bool = True,
) -> dict:
    """定位多张圆环

    Args:
        images (list[np.ndarray]): 多张图片
        roi_names: (list[int | str]): 每张图片对应的名称
        timestamp: Any | None: 时间戳. Defaults to None.
        subpixel_edges_threshold (int, optional): 子像素梯度阈值. Defaults to 25.
        iters (int, optional): 子像素检测迭代次数，模糊图像可以增加迭代次数. Defaults to 0.
        order (int, optional): 边缘检测使用公式的阶数. Defaults to 2.
        rings_nums (int, optional): 要检测的圆环的数量. Defaults to 6.
        min_group_size (int, optional): 每个圆环最小的点数,小于这个值的圆的组会被丢弃. Defaults to 3.
        sigmas (list[float | int] | float | int, optional): 拟合圆时通过拟合的圆心到每个点的距离进行过滤，是通过误差的正态分布过滤的，<=0 代表禁用过滤. Defaults to 0.
            这个值的含义是 n 个 sigma 的值，有几个值，就过滤几次，每个值代表本次过滤的 sigma 的值
            - 横轴区间（μ-σ,μ+σ）内的面积为68.268949%
            - 横轴区间（μ-2σ,μ+2σ）内的面积为95.449974%
            - 横轴区间（μ-3σ,μ+3σ）内的面积为99.730020%
        save_dir (str, optional): 保存图片路径. Defaults to "./".
        draw_scale (int, optional): 缩放图片的倍率. Defaults to 20.
        save_grads (bool, optional): 是否保存梯度图. Defaults to False.
        save_detect_images (bool, optional): 是否保存文件. Defaults to True.
        save_detect_results (bool, optional): 是否保存检测结果. Defaults to True.

    Returns:
        dict: 检测结果
    """
    timestamp = (
        time.strftime("%Y%m%d-%H%M%S", time.localtime())
        if timestamp is None
        else timestamp
    )
    assert len(images) == len(
        roi_names
    ), "images number must equal to roi_images number"

    measure_results = {}
    for image, roi_name in zip(images, roi_names):
        measure_results[roi_name] = rings_location(
            image=image,
            timestamp=f"{timestamp}-{roi_name}",
            subpixel_edges_threshold=subpixel_edges_threshold,
            iters=iters,
            order=order,
            rings_nums=rings_nums,
            min_group_size=min_group_size,
            sigmas=sigmas,
            save_dir=save_dir,
            draw_scale=draw_scale,
            save_grads=save_grads,
            save_detect_images=save_detect_images,
            save_detect_results=save_detect_results,
        )
    return measure_results


def adaptive_threshold_rings_location(
    image: np.ndarray,
    timestamp: Any | None = None,
    iters: int = 0,
    order: int = 2,
    rings_nums: int = 6,
    min_group_size: int = 3,
    sigmas: list[float | int] | float | int = 0,
    save_dir: str | Path = "./save_dir",
    draw_scale: int = 20,
    save_grads: bool = False,
    save_detect_images: bool = True,
    save_detect_results: bool = True,
    gradient_threshold_percent: float = 0.6,
) -> dict:
    """动态梯度阈值定位圆环

    Args:
        image (np.ndarray): 图片
        timestamp: Any | None: 时间戳. Defaults to None.
        iters (int, optional): 子像素检测迭代次数，模糊图像可以增加迭代次数. Defaults to 0.
        order (int, optional): 边缘检测使用公式的阶数. Defaults to 2.
        rings_nums (int, optional): 要检测的圆环的数量. Defaults to 6.
        min_group_size (int, optional): 每个圆环最小的点数,小于这个值的圆的组会被丢弃. Defaults to 3.
        sigmas (list[float | int] | float | int, optional): 拟合圆时通过拟合的圆心到每个点的距离进行过滤，是通过误差的正态分布过滤的，<=0 代表禁用过滤. Defaults to 0.
            这个值的含义是 n 个 sigma 的值，有几个值，就过滤几次，每个值代表本次过滤的 sigma 的值
            - 横轴区间（μ-σ,μ+σ）  内的面积为68.268949%
            - 横轴区间（μ-2σ,μ+2σ）内的面积为95.449974%
            - 横轴区间（μ-3σ,μ+3σ）内的面积为99.730020%
        save_dir (str, optional): 保存图片路径. Defaults to "./".
        draw_scale (int, optional): 缩放图片的倍率. Defaults to 20.
        save_grads (bool, optional): 是否保存梯度图. Defaults to False.
        save_detect_images (bool, optional): 是否保存文件. Defaults to True.
        save_detect_results (bool, optional): 是否保存检测结果. Defaults to True.
        gradient_threshold_percent (float, optional): 梯度图的阈值，取值范围 0-1. Defaults to 0.6.

    Returns:
        dict: 检测结果
    """

    # 根据梯度获取阈值
    _, grad_crop = image_gradient(image, iters=iters)
    threshold = get_gradient_threshold(grad_crop, gradient_threshold_percent)

    result = rings_location(
        image=image,
        timestamp=timestamp,
        subpixel_edges_threshold=threshold,
        iters=iters,
        order=order,
        rings_nums=rings_nums,
        min_group_size=min_group_size,
        sigmas=sigmas,
        save_dir=save_dir,
        draw_scale=draw_scale,
        save_grads=save_grads,
        save_detect_images=save_detect_images,
        save_detect_results=save_detect_results,
    )
    return result


if __name__ == "__main__":
    image: np.ndarray = cv2.imread("images/circles2-7.5cm-400.png")
    print(image.shape)

    result = adaptive_threshold_rings_location(
        image=image,
        timestamp="circles2-7.5cm-500",
        iters=1,
        order=2,
        rings_nums=6,
        min_group_size=3,
        sigmas=2,
        save_dir="./save_dir",
        draw_scale=20,
        save_grads=False,
        save_detect_images=True,
        save_detect_results=True,
        gradient_threshold_percent=0.6,
    )
    print(result)
