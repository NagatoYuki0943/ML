import numpy as np
from copy import deepcopy
from loguru import logger


def fit_circle_by_least_square0(
    points: list | np.ndarray,
) -> tuple[float, float, float, float, float, float, float, np.ndarray]:
    """根据最小二乘法求圆心和半径

    https://www.cnblogs.com/xiaxuexiaoab/p/16276402.html

    https://kimi.moonshot.cn/chat/cp290dsubms6ha9m74j0
    Args:
        points (list | np.ndarray): 圆的边缘的xy值, 二维数据
            example:
                [[x1, y1], [x2, y2], [x3, y3]...]

    Returns:
        tuple: 中心坐标x, 中心坐标y, 半径, 计算标准差
    """
    points = np.array(points)
    N = points.shape[0]
    assert N >= 3, f"圆的边缘点数量至少为3，当前边缘点数量为{N}"

    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_x3 = 0.0
    sum_y3 = 0.0
    sum_xy = 0.0
    sum_xy2 = 0.0
    sum_x2y = 0.0

    for x, y in points:
        sum_x += x
        sum_y += y

        x2 = x * x
        y2 = y * y
        sum_x2 += x2
        sum_y2 += y2

        sum_x3 += x2 * x
        sum_y3 += y2 * y

        sum_xy += x * y

        sum_xy2 += x * y2
        sum_x2y += x2 * y

    C = N * sum_x2 - sum_x**2
    D = N * sum_xy - sum_x * sum_y
    E = N * sum_x3 + N * sum_xy2 - (sum_x2 + sum_y2) * sum_x
    G = N * sum_y2 - sum_y**2
    H = N * sum_x2y + N * sum_y3 - (sum_x2 + sum_y2) * sum_y

    a = (H * D - E * G) / (C * G - D * D)
    b = (H * C - E * D) / (D * D - G * C)
    c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N

    center_x = -a / 2.0
    center_y = -b / 2.0
    radius = np.sqrt(a**2 + b**2 - 4 * c) / 2.0

    # err = 0.0
    # for point in points:
    #     e = np.sum((point - np.array([center_x, center_y])) ** 2) - radius ** 2
    #     if e > err:
    #         err = e
    # 求每个点到求出来的中心的距离
    points_to_center = np.sqrt(
        np.sum((points - np.array([center_x, center_y])) ** 2, axis=1)
    )
    # 计算中心距离和求出来的半径的差距
    radii_err = points_to_center - radius
    # 差距的均值
    err_avg = np.mean(radii_err)
    # 差距的方差
    err_var = np.var(radii_err)
    # 差距的标准差
    err_std = np.std(radii_err)
    # 差距的绝对差距
    err_abs = np.abs(radii_err)

    return center_x, center_y, radius, err_avg, err_var, err_std, err_abs, radii_err


def fit_circle_by_least_square(
    points: list | np.ndarray,
) -> tuple[float, float, float, float, float, float, float, np.ndarray]:
    """根据最小二乘法求圆心和半径

    Args:
        points (list | np.ndarray): 圆的边缘的xy值, 二维数据
            example:
                [[x1, y1], [x2, y2], [x3, y3]...]

    Returns:
        tuple: 中心坐标x, 中心坐标y, 半径, 计算标准差
    """
    points = np.array(points)
    N = points.shape[0]
    assert N >= 3, f"圆的边缘点数量至少为3，当前边缘点数量为{N}"

    x = points[:, 0]
    y = points[:, 1]
    x2 = x**2
    y2 = y**2

    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = x2.sum()
    sum_y2 = y2.sum()
    sum_x3 = (x2 * x).sum()
    sum_y3 = (y2 * y).sum()
    sum_xy = (x * y).sum()
    sum_xy2 = (x * y2).sum()
    sum_x2y = (x2 * y).sum()

    C = N * sum_x2 - sum_x**2
    D = N * sum_xy - sum_x * sum_y
    E = N * sum_x3 + N * sum_xy2 - (sum_x2 + sum_y2) * sum_x
    G = N * sum_y2 - sum_y**2
    H = N * sum_x2y + N * sum_y3 - (sum_x2 + sum_y2) * sum_y

    a = (H * D - E * G) / (C * G - D * D)
    b = (H * C - E * D) / (D * D - G * C)
    c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N

    center_x = -a / 2.0
    center_y = -b / 2.0
    radius = np.sqrt(a**2 + b**2 - 4 * c) / 2.0

    # 求每个点到求出来的中心的距离
    points_to_center = np.sqrt(
        np.sum((points - np.array([center_x, center_y])) ** 2, axis=1)
    )
    # 计算中心距离和求出来的半径的差距
    radii_err = points_to_center - radius
    # 差距的均值
    err_avg = np.mean(radii_err)
    # 差距的方差
    err_var = np.var(radii_err)
    # 差距的标准差
    err_std = np.std(radii_err)
    # 差距的绝对差距
    err_abs = np.abs(radii_err)

    return center_x, center_y, radius, err_avg, err_var, err_std, err_abs, radii_err


def fit_circle_by_least_square_filter(
    points: list | np.ndarray, sigmas: list[float | int] | float | int = 0
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    None | np.ndarray,
]:
    """通过求解出来的圆心点到每个真实点的距离，与求解出来的半径的标准差进行过滤

    Args:
        points (list | np.ndarray): 圆边缘坐标
        sigmas (list[float | int] | float | int, optional): 拟合圆时通过拟合的圆心到每个点的距离进行过滤，是通过误差的正态分布过滤的，<=0 代表禁用过滤. Defaults to 0.
        这个值的含义是 n 个 sigma 的值，有几个值，就过滤几次，每个值代表本次过滤的 sigma 的值
            - 横轴区间（μ-σ,μ+σ）内的面积为68.268949%
            - 横轴区间（μ-2σ,μ+2σ）内的面积为95.449974%
            - 横轴区间（μ-3σ,μ+3σ）内的面积为99.730020%

    Returns:
        tuple: results
    """
    # 第一次执行
    center_x, center_y, radius, err_avg, err_var, err_std, err_abs, radii_err = (
        fit_circle_by_least_square(points)
    )

    need_wrap = not isinstance(sigmas, list)
    sigmas = [sigmas] if need_wrap else sigmas

    # 过滤偏移点
    fit_circle: list | np.ndarray = deepcopy(points)

    ignore_circle = None
    for i, sigma in enumerate(sigmas):
        if sigma <= 0:
            # sigma <= 0,代表不过滤
            continue
        # 根据正态分布过滤偏离点
        n_sigma = err_std * sigma
        # 先找忽略的，再找保留的，因为保留的数据会覆盖原数据
        # 要使用 or 不用 not
        ignore_circle: np.ndarray = fit_circle[
            np.bitwise_or(
                radii_err <= err_avg - n_sigma, radii_err >= err_avg + n_sigma
            )
        ]
        fit_circle: np.ndarray = fit_circle[
            np.bitwise_and(radii_err > err_avg - n_sigma, radii_err < err_avg + n_sigma)
        ]
        points_num: int = fit_circle.shape[0]
        if points_num >= 3:
            (
                center_x,
                center_y,
                radius,
                err_avg,
                err_var,
                err_std,
                err_abs,
                radii_err,
            ) = fit_circle_by_least_square(fit_circle)
        else:
            logger.error(f"用于拟合的圆的数量为{points_num}，无法过滤，跳过过滤步骤")
            break

    return (
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
    )
