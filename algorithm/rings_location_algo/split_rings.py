import numpy as np
from loguru import logger

from .fit_circle_by_least_square import fit_circle_by_least_square


def split_rings(
    points: np.ndarray,
    rings_nums: int,
    threshold_range: float = 0.5,
    min_group_size: int = 0,
    momentum: float = 0.9,
) -> list:
    """将同心圆环坐标分为多个独立的圆环
    assist by kimi

    Args:
        points (np.ndarray): xy坐标 [[x1, y1], [x2, y2], ...]
        rings_nums (int, optional): 圆环数量.
        threshold_range (float, optional): 设置阈值，这里我们取相邻同心圆半径差的一半. Defaults to 0.5.
        min_group_size (int, optional): 分组时每个类别的最小数量，会过滤掉分组数小于这个值的坐标，如果不想过滤，将这个值调整为<=0即可. Defaults to 0.
        momentum (float, optional): 通过动量动态更新⚪的半径. Defaults to 0.9.

    Returns:
        list: 分组的坐标
    """
    assert momentum >= 0 and momentum <= 1, f"momentum should be between 0 and 1, got {momentum}"

    # 中心点
    # center_x, center_y = np.mean(points, axis=0)
    results = fit_circle_by_least_square(points)
    center_x = results[0]
    center_y = results[1]

    # 每个点到中心点的距离
    distances = np.sqrt(np.sum((points - np.array([center_x, center_y])) ** 2, axis=1))
    # 距离的升序排序
    distances_sort = np.sort(distances)
    # 距离的升序排序id
    distances_sort_index = np.argsort(distances)
    # 距离的范围
    distances_range = distances.max() - distances.min()

    # 设置阈值，这里我们取相邻同心圆半径差的一半，例如(10-9)/2 = 0.5
    threshold = distances_range / (rings_nums - 1) * threshold_range

    # 用于存储每个分组的半径
    group_radii = []

    # 初始化分组列表
    group_indexes = []

    # 遍历每个距离
    for i, distance in enumerate(distances_sort):
        # 检查是否已经属于某个分组
        assigned = False
        for j, radius in enumerate(group_radii):
            # 如果距离与分组半径相差小于阈值，则分配到该分组
            if abs(distance - radius) <= threshold:
                # 存储的是距离排序的id,对应没有排序前的id
                group_indexes[j].append(distances_sort_index[i])

                # 通过新的属于同一个组的值更新原来的 radius
                group_radii[j] = momentum * radius + (1 - momentum) * distance
                assigned = True
                break

        # 如果没有分配，则创建新的分组
        if not assigned:
            group_radii.append(distance)
            # 存储的是距离排序的id,对应没有排序前的id
            group_indexes.append([distances_sort_index[i]])

    group_rings = []
    # 输出分组结果
    for i, group_index in enumerate(group_indexes):
        if len(group_index) < min_group_size:
            # logger.info(f"ignore {len(group_index)} points")
            continue
        group_rings.append(points[group_index])
    return group_rings


def split_rings_adaptive(
    points: np.ndarray,
    rings_nums: int,
    min_group_size: int = 0,
    momentum: float = 0.9,
    init_threshold_range: float = 0.5,
    range_change: float = 0.01,
    times: int = 100
) -> list:
    """找到的圆环数量如果大于指定的数量，要调高 threshold_range，否则减小 threshold_range

    Args:
        points (np.ndarray): xy坐标 [[x1, y1], [x2, y2], ...]
        rings_nums (int, optional): 圆环数量.
        min_group_size (int, optional): 分组时每个类别的最小数量，会过滤掉分组数小于这个值的坐标，如果不想过滤，将这个值调整为<=0即可. Defaults to 0.
        init_threshold_range (float, optional): 初始设置阈值，这里我们取相邻同心圆半径差的一半. Defaults to 0.5.
        range_change (float, optional): 每次变换阈值的步长. Defaults to 0.01
        times (int, optional): 循环次数. Defaults to 100.
        momentum (float, optional): 通过动量动态更新⚪的半径. Defaults to 0.9.

    Returns:
        list: 分组的坐标
    """
    threshold_range = init_threshold_range
    for i in range(times):
        # 检测圆环
        group_rings = split_rings(points, rings_nums, threshold_range, min_group_size, momentum)
        # 检测到的圆环数量
        detect_rings_nums = len(group_rings)
        if detect_rings_nums == 0:
            raise ValueError("没有检测到圆环")

        # logger.info(f"try time {i + 1}: {threshold_range = }, {detect_rings_nums = }")
        if detect_rings_nums == rings_nums:
            # 数量相等,就返回
            return group_rings
        elif detect_rings_nums > rings_nums:
            # 检测圆环数量多，提高阈值
            threshold_range += range_change
        elif detect_rings_nums < rings_nums:
            # 检测圆环数量少，降低阈值
            threshold_range -= range_change

        # 如果阈值小于等于0，就直接返回最后一次的检测结果
        if threshold_range <= 0:
            logger.warning("threshold_range <= 0, use last result")
            return group_rings

    raise ValueError(f"找不到 {rings_nums = } 数量的圆环")
