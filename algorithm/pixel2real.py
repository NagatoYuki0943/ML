"""
imaging_size: 成像大小
focus: 焦距
object_distance: 物距
object_size: 物体大小
pixel_num: 像素数量
pixel_size: 像元大小

\frac {imaging\_size} {focus} = \frac {object\_distance} {object\_size}

imaging\_size = pixel\_num * pixel\_size
"""

import numpy as np


def pixel_num2object_distance(
    pixel_num: float | np.ndarray,
    pixel_size: float | np.ndarray = 1.55,
    focus: float | np.ndarray = 8,
    object_size: float | np.ndarray = 65,
) -> float | np.ndarray:
    """根据像素大小, 焦距, 物体大小反推出物距.

    Args:
        pixel_num (float | np.ndarray): 像素数量.
        pixel_size (float | np.ndarray), optional): 单个像素大小, 默认为1.55um. Defaults to 1.55.
        focus (float | np.ndarray), optional): 焦距, 默认为8mm. Defaults to 8.
        object_size (float | np.ndarray), optional): 物体大小, 默认为65mm. Defaults to 65.

    Returns:
        float | np.ndarray: 物距, 单位为mm.
    """
    imaging_size = pixel_num * pixel_size / 1000  # 单位为mm
    object_distance = focus * object_size / imaging_size  # 单位为mm
    return object_distance


def test_pixel_num2object_distance():
    # pixel_num, pixel_size, focus, object_size
    params = [
        [20, 1.65, 8, 65],
        [20.8, 1.65, 8, 65],
        [21, 1.65, 8, 65],
        [21.2, 1.65, 8, 65],
        [22, 1.65, 8, 65],
    ]

    for pixel_num, pixel_size, focus, object_size in params:
        object_distance = pixel_num2object_distance(
            pixel_num,
            pixel_size,
            focus,
            object_size,
        )
        print(object_distance)
    # 15757.575757575756
    # 15151.51515151515
    # 15007.215007215007
    # 14865.637507146943
    # 14325.068870523417

    pixel_nums = np.array([20, 20.8, 21, 21.2, 22])
    pixel_num2object_distances = pixel_num2object_distance(
        pixel_nums,
        1.65,
        8,
        65,
    )
    print(pixel_num2object_distances)
    # [15757.57575758 15151.51515152 15007.21500722 14865.63750715 14325.06887052]


def pixel_num2object_size(
    pixel_num: float | np.ndarray,
    object_distance: float | np.ndarray,
    pixel_size: float | np.ndarray = 1.55,
    focus: float | np.ndarray = 8,
) -> float | np.ndarray:
    """根据物体像素大小,焦距,物距反推出物体大小.

    Args:
        pixel_num (float | np.ndarray): 像素数量.
        object_distance (float | np.ndarray): 物距.
        pixel_size (float | np.ndarray, optional): 单个像素大小, 默认为1.55um. Defaults to 1.55.
        focus (float | np.ndarray, optional): 焦距, 默认为8mm. Defaults to 8.

    Returns:
        float | np.ndarray: 物体大小, 单位为mm.
    """
    imaging_size = pixel_num * pixel_size / 1000  # 单位为mm
    object_size = imaging_size * object_distance / focus  # 单位为mm
    return object_size


def test_pixel_num2object_size():
    # pixel_num, object_distance, pixel_size, focus
    params = [
        [20, 15757, 1.65, 8],
        [20.8, 15151, 1.65, 8],
        [21, 15007, 1.65, 8],
        [21.2, 14865, 1.65, 8],
        [22, 14325, 1.65, 8],
    ]

    for pixel_num, object_distance, pixel_size, focus in params:
        object_distance = pixel_num2object_size(
            pixel_num,
            object_distance,
            pixel_size,
            focus,
        )
        print(object_distance)
    # 64.997625
    # 64.99779000000001
    # 64.99906875
    # 64.99721249999999
    # 64.9996875

    pixel_nums = np.array([20, 20.8, 21, 21.2, 22])
    object_distances = np.array([15757, 15151, 15007, 14865, 14325])
    pixel_num2object_distances = pixel_num2object_size(
        pixel_nums,
        object_distances,
        1.65,
        8,
    )
    print(pixel_num2object_distances)
    # [64.997625   64.99779    64.99906875 64.9972125  64.9996875 ]


if __name__ == "__main__":
    test_pixel_num2object_distance()
    print("=" * 50)
    test_pixel_num2object_size()
