import cv2
import numpy as np
from loguru import logger


class StereoCalibration:
    def __init__(
        self,
        K: np.ndarray | list[list[float]] | None = None,
        dist_coeffs: np.ndarray | list[list[float]] | None = None,
    ):
        """初始化图像矫正器

        Args:
            K (np.ndarray | list[list[float]]): 相机内参矩阵 (3x3)
            dist_coeffs (np.ndarray | list[list[float]]): 畸变系数 (k1, k2, p1, p2, k3)
        """
        self.K = np.array(K) if K is not None else None
        self.dist_coeffs = np.array(dist_coeffs) if dist_coeffs is not None else None

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """图像畸变矫正"""
        if any([self.K is None, self.dist_coeffs is None]):
            logger.warning("K or dist_coeffs is None, couldn't undistort image")
            return image

        undistorted_img: np.ndarray = cv2.undistort(image, self.K, self.dist_coeffs)
        return undistorted_img

    def undistort_point(self, point: np.ndarray | list[float]) -> list[float]:
        """对单个点坐标进行畸变矫正

        Args:
            point (np.ndarray | list[float]): 需要矫正的坐标, 格式为 [x, y]

        Returns:
            list[float]: 矫正后的坐标, 格式为 [x, y]
        """
        # 将点转换为符合输入格式的数组
        point = np.array(point)

        if any([self.K is None, self.dist_coeffs is None]):
            logger.warning("K or dist_coeffs is None, couldn't undistort point")
            return point.tolist()

        # 执行点的畸变矫正
        undistorted_point: np.ndarray = cv2.undistortPoints(
            point, self.K, self.dist_coeffs, P=self.K
        )

        # 返回矫正后的点坐标
        point_corrected: list = undistorted_point.ravel().tolist()
        return point_corrected

    def undistort_points(self, points: np.ndarray | list[list[float]]) -> list[list[float]]:
        """对多个点坐标进行畸变矫正

        Args:
            points (np.ndarray | list[list[float]]): 多个坐标, 格式为 [[x1, y1], [x2, y2], ...]

        Returns:
            list[list[float]]: 矫正后的坐标, 格式为 [[x1, y1], [x2, y2], ...]
        """
        points = np.array(points).reshape(-1, 2)

        if any([self.K is None, self.dist_coeffs is None]):
            logger.warning("K or dist_coeffs is None, couldn't undistort point")
            return points.tolist()

        undistorted_points = []
        for point in points:
            undistorted_point: np.ndarray = cv2.undistortPoints(
                point, self.K, self.dist_coeffs, P=self.K
            )
            point_corrected: list = undistorted_point.ravel().tolist()
            undistorted_points.append(point_corrected)
        return undistorted_points

    # def undistort_point(self, u, v):
    #     """
    #     对单个点坐标进行畸变矫正
    #     :param u: 原始点的x坐标
    #     :param v: 原始点的y坐标
    #     :return: 矫正后的点坐标 (u_corrected, v_corrected)
    #     """
    #     # 将点转换为符合输入格式的数组
    #     distorted_point = np.array([[[u, v]]], dtype="float32")

    #     # 执行点的畸变矫正
    #     undistorted_point = cv2.undistortPoints(
    #         distorted_point, self.K, self.dist_coeffs, P=self.K
    #     )

    #     # 返回矫正后的点坐标
    #     u_corrected, v_corrected = undistorted_point[0][0]
    #     return u_corrected, v_corrected


if __name__ == "__main__":
    # 相机内参矩阵 (K) 和畸变系数 (dist_coeffs)
    K = np.array(
        [
            [5.19269425e03, 0.00000000e00, 2.02199831e03],
            [0.00000000e00, 5.19320469e03, 1.56584753e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    dist_coeffs = np.array(
        [
            [
                -8.41673780e-02,
                -9.15387560e-02,
                9.50935715e-04,
                9.70130486e-05,
                1.48280474e00,
            ]
        ]
    )  # 畸变系数

    # 创建ImageUndistorter对象
    stereo_calibration = StereoCalibration(K, dist_coeffs)

    # 输入图像路径
    image_path = "../assets/template/2circles/2circles-6_5-3-500pixel.png"
    image = cv2.imread(image_path)
    print(image.shape)

    # 输出图像路径
    output_path = "../results/2circles-6_5-3-500pixel.jpg"

    # 调用函数进行图像畸变矫正
    undistorted_image = stereo_calibration.undistort_image(image)

    cv2.imwrite(output_path, undistorted_image)

    # 这里进行模板匹配中心点识别
    # 输入需要矫正的点坐标
    xy = [200.0, 200.0]
    # 畸变矫正后的坐标点
    xy_corrected = stereo_calibration.undistort_point(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")

    xy = [[200.0, 200.0]]
    # 畸变矫正后的坐标点
    xy_corrected = stereo_calibration.undistort_point(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")

    xy = [[[200.0, 200.0]]]
    # 畸变矫正后的坐标点
    xy_corrected = stereo_calibration.undistort_point(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")

    xy = [[200.0, 200.0], [400.0, 400.0]]
    # 畸变矫正后的坐标点
    xy_corrected = stereo_calibration.undistort_points(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")

    print("=" * 100)
    # 测试为None的情况
    stereo_calibration = StereoCalibration()
    undistorted_image = stereo_calibration.undistort_image(image)
    xy = [200.0, 200.0]
    xy_corrected = stereo_calibration.undistort_point(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")
    xy = [[200.0, 200.0], [400.0, 400.0]]
    xy_corrected = stereo_calibration.undistort_points(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")
