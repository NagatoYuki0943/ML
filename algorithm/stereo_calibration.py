import cv2
import numpy as np


class StereoCalibration:
    def __init__(
        self,
        K: np.ndarray | list[list[float]],
        dist_coeffs: np.ndarray | list[list[float]],
    ):
        """初始化图像矫正器

        Args:
            K (np.ndarray | list[list[float]]): 相机内参矩阵 (3x3)
            dist_coeffs (np.ndarray | list[list[float]]): 畸变系数 (k1, k2, p1, p2, k3)
        """
        self.K = np.array(K)
        self.dist_coeffs = np.array(dist_coeffs)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """执行图像畸变矫正"""
        undistorted_img = cv2.undistort(image, self.K, self.dist_coeffs)
        return undistorted_img

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

    def undistort_point(self, xy: np.ndarray | list[float]) -> np.ndarray:
        """对单个点坐标进行畸变矫正

        Args:
            xy (np.ndarray | list[float]): 需要矫正的坐标

        Returns:
            ndarray: 矫正后的坐标
        """
        # 将点转换为符合输入格式的数组
        # distorted_point = np.array([[xy]])
        distorted_point = np.array(xy)

        # 执行点的畸变矫正
        undistorted_point = cv2.undistortPoints(
            distorted_point, self.K, self.dist_coeffs, P=self.K
        )

        # 返回矫正后的点坐标
        xy_corrected: np.ndarray = undistorted_point[0][0]
        return xy_corrected.tolist()


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
    undistorter = StereoCalibration(K, dist_coeffs)

    # 输入图像路径
    image_path = "../results/image0_default.jpg"
    image = cv2.imread(image_path)

    # 输出图像路径
    output_path = "../results/image0_undistorted.jpg"

    # 调用函数进行图像畸变矫正
    undistorted_image = undistorter.undistort_image(image)

    cv2.imwrite(output_path, undistorted_image)

    # 这里进行模板匹配中心点识别
    # 输入需要矫正的点坐标
    xy = [1000.0, 1000.0]

    # 畸变矫正后的坐标点
    xy_corrected = undistorter.undistort_point(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")
