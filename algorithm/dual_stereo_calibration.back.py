import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from loguru import logger


class DualStereoCalibration:
    def __init__(
        self,
        camera_matrix_left: np.ndarray | list[list[float]] | None = None,
        camera_matrix_right: np.ndarray | list[list[float]] | None = None,
        distortion_coefficients_left: np.ndarray | list[list[float]] | None = None,
        distortion_coefficients_right: np.ndarray | list[list[float]] | None = None,
        R: np.ndarray | list[list[float]] | None = None,
        T: np.ndarray | list[list[float]] | None = None,
        pixel_width_mm: float | None = None,
    ):
        self.camera_matrix_left = (
            np.array(camera_matrix_left) if camera_matrix_left is not None else None
        )
        self.camera_matrix_right = (
            np.array(camera_matrix_right) if camera_matrix_right is not None else None
        )
        self.distortion_coefficients_left = (
            np.array(distortion_coefficients_left)
            if distortion_coefficients_left is not None
            else None
        )
        self.distortion_coefficients_right = (
            np.array(distortion_coefficients_right)
            if distortion_coefficients_right is not None
            else None
        )
        self.R = np.array(R) if R is not None else None
        self.T = np.array(T) if T is not None else None
        self.pixel_width_mm = pixel_width_mm if pixel_width_mm is not None else None

    def undistort_images(
        self, image_left: np.ndarray, image_right: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        tuple | None,
        tuple | None,
    ]:
        """图像畸变矫正"""
        if any(
            [
                self.camera_matrix_left is None,
                self.camera_matrix_right is None,
                self.distortion_coefficients_left is None,
                self.distortion_coefficients_right is None,
                self.R is None,
                self.T is None,
            ]
        ):
            logger.warning("camera matrices not set, couldn't undistort images.")
            return (
                image_left,
                image_right,
                image_left,
                image_right,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # w, h
        image_size = (image_left.shape[1], image_left.shape[0])

        # focal_length_pixels_R = self.camera_matrix_right[0, 0]
        # focal_length_pixels_L = self.camera_matrix_left[0, 0]
        # logger.info(f"Focal length (in pixels): {focal_length_pixels}")
        # 给定的传感器尺寸和图像分辨率
        # sensor_width_mm = 6.413  # 传感器宽度，以毫米为单位
        # image_width_pixels = 3840  # 图像宽度，以像素为单位

        # 计算每个像素的宽度（以毫米为单位）
        # pixel_width_mm = sensor_width_mm / image_width_pixels

        # 将焦距从像素转换为毫米
        # focal_length_mm_R = focal_length_pixels_R * self.pixel_width_mm
        # logger.info("R origin focal", focal_length_mm_R)
        # focal_length_mm_L = focal_length_pixels_L * self.pixel_width_mm
        # logger.info("R origin focal", focal_length_mm_L)

        # 获取立体校正参数
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.camera_matrix_left,
            None,
            self.camera_matrix_right,
            None,
            image_size,
            self.R,
            self.T,
        )

        # 使用 cv2.undistort 进行畸变矫正
        undistorted_left: np.ndarray = cv2.undistort(
            image_left, self.camera_matrix_left, self.distortion_coefficients_left
        )
        undistorted_right: np.ndarray = cv2.undistort(
            image_right, self.camera_matrix_right, self.distortion_coefficients_right
        )

        # 计算校正映射
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, None, R1, P1, image_size, cv2.CV_32FC1
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, None, R2, P2, image_size, cv2.CV_32FC1
        )

        # 应用校正映射
        rectified_image_left = cv2.remap(
            undistorted_left, map1_left, map2_left, cv2.INTER_LINEAR
        )
        rectified_image_right = cv2.remap(
            undistorted_right, map1_right, map2_right, cv2.INTER_LINEAR
        )

        return (
            rectified_image_left,
            rectified_image_right,
            undistorted_left,
            undistorted_right,
            R1,
            R2,
            P1,
            P2,
            Q,
            roi1,
            roi2,
        )

    def undistort_point(
        self,
        point: np.ndarray | list[float],
        camera_matrix: np.ndarray | None,
        R: np.ndarray | None,
        P: np.ndarray | None,
    ) -> list[float]:
        """对单个点坐标进行畸变矫正

        Args:
            point (np.ndarray | list[float]): 需要矫正的坐标, 格式为 [x, y]
            camera_matrix (np.ndarray):
            R (np.ndarray):
            P (np.ndarray):

        Returns:
            list[float]: 矫正后的坐标, 格式为 [x, y]
        """

        # 将点转换为符合输入格式的数组
        point = np.array(point)

        if any([camera_matrix is None, R is None, P is None]):
            logger.warning("camera matrices not set, couldn't undistort points.")
            return point.tolist()

        undistorted_point: np.ndarray = cv2.undistortPoints(
            point, camera_matrix, None, R=R, P=P
        )
        # 返回矫正后的点坐标
        point_corrected: list = undistorted_point.ravel().tolist()
        return point_corrected

    def undistort_points(
        self,
        points: np.ndarray | list[list[float]],
        camera_matrix: np.ndarray | None,
        R: np.ndarray | None,
        P: np.ndarray | None,
    ) -> list[list[float]]:
        """对多个点坐标进行畸变矫正

        Args:
            points (np.ndarray | list[list[float]]): 多个坐标, 格式为 [[x1, y1], [x2, y2], ...]
            camera_matrix (np.ndarray):
            R (np.ndarray):
            P (np.ndarray):

        Returns:
            list[list[float]]: 矫正后的坐标, 格式为 [[x1, y1], [x2, y2], ...]
        """
        points = np.array(points).reshape(-1, 2)

        if any([camera_matrix is None, R is None, P is None]):
            logger.warning("camera matrices not set, couldn't undistort points.")
            return points.tolist()

        undistorted_points = []
        for point in points:
            undistorted_point: np.ndarray = cv2.undistortPoints(
                point, camera_matrix, None, R=R, P=P
            )
            point_corrected: list = undistorted_point.ravel().tolist()
            undistorted_points.append(point_corrected)
        return undistorted_points

    def find_corners(self, rectified_image: np.ndarray, chessboard_size=(11, 8)):
        gray_image = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray_image, chessboard_size, None)
        if ret:
            corners = corners.reshape(-1, 2)
        return ret, corners

    def save_corners_to_csv(self, corners, file_path):
        df = pd.DataFrame(corners, columns=["x", "y"])
        df.to_csv(file_path, index=False)

    def get_stereo_parameters(self, P1, P2):
        # 焦距是投影矩阵中的元素
        focal_length_pixels = P1[0, 0]
        # logger.info(f"Focal length (in pixels): {focal_length_pixels}")

        # 基线距离可以从 P2 的第四列第三个元素计算得到
        baseline = -P2[0, 3] / P2[0, 0]
        logger.info(f"Baseline (in meters): {baseline}")

        # 给定的传感器尺寸和图像分辨率
        # sensor_width_mm = 6.413  # 传感器宽度，以毫米为单位
        # image_width_pixels = 3840  # 图像宽度，以像素为单位

        # 计算每个像素的宽度（以毫米为单位）
        # pixel_width_mm = sensor_width_mm / image_width_pixels

        # 将焦距从像素转换为毫米
        focal_length_mm = focal_length_pixels * self.pixel_width_mm
        logger.info(f"Focal length (in millimeters): {focal_length_mm:.2f} mm")

        return focal_length_pixels, baseline

    def pixel_to_world(self, P1, P2, left_points, right_points):
        left_world_points = []
        right_world_points = []
        disparities = []
        depths = []

        # 获取焦距和基线距离
        focal_length, baseline = self.get_stereo_parameters(P1, P2)

        # 将点转换为 NumPy 数组并转置
        left_points_array = np.array(left_points).reshape(-1, 2).T
        right_points_array = np.array(right_points).reshape(-1, 2).T

        # 将点转换为齐次坐标
        points4D = cv2.triangulatePoints(P1, P2, left_points_array, right_points_array)

        # 转换为非齐次坐标
        points3D = points4D[:3] / points4D[3]

        for i in range(points3D.shape[1]):
            left_point = left_points[i]
            right_point = right_points[i]
            disparity = left_point[0] - right_point[0]
            if disparity == 0:
                continue
            disparities.append(disparity)

            # 根据视差计算深度
            depth = (focal_length * baseline) / disparity
            depths.append(depth)

            left_world_point = points3D[:, i]
            left_world_points.append(left_world_point[:3])

            # Note: Right world points are the same as left in this context,
            # as triangulatePoints returns a single 3D point for each pair of 2D points.
            right_world_points.append(left_world_point[:3])

        avg_disparity = np.mean(disparities) if disparities else 0
        avg_depth = np.mean(depths) if depths else 0

        return (
            np.array(left_world_points),
            np.array(right_world_points),
            avg_disparity,
            avg_depth,
            depths,
            focal_length,
        )

    def draw_combined_image(
        self, rectified_image_left, rectified_image_right, output_path
    ):
        # logger.info(rectified_image_left.shape)
        height, width = rectified_image_left.shape

        # Convert grayscale images to 3-channel images
        rectified_image_left_3ch = cv2.cvtColor(
            rectified_image_left, cv2.COLOR_GRAY2BGR
        )
        rectified_image_right_3ch = cv2.cvtColor(
            rectified_image_right, cv2.COLOR_GRAY2BGR
        )

        combined_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
        combined_image[:, :width] = rectified_image_left_3ch
        combined_image[:, width:] = rectified_image_right_3ch

        combined_image_pil = Image.fromarray(
            cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        )

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(combined_image_pil)
        for i in range(1, 20):
            y = i * height // 20
            ax.axhline(y=y, color="r", linestyle="-")

        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


if __name__ == "__main__":
    camera_matrix_left = np.array(
        [
            [7.44937603e03, 0.00000000e00, 1.79056889e03],
            [0.00000000e00, 7.45022891e03, 1.26665786e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    camera_matrix_right = np.array(
        [
            [7.46471035e03, 0.00000000e00, 1.81985040e03],
            [0.00000000e00, 7.46415680e03, 1.38081032e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    distortion_coefficients_left = np.array(
        [
            [
                -4.44924086e-01,
                6.27814725e-01,
                -1.80510014e-03,
                -8.97545764e-04,
                -1.84473439e01,
            ]
        ]
    )
    distortion_coefficients_right = np.array(
        [
            [
                -4.07660445e-01,
                -2.23391154e00,
                -1.09115383e-03,
                -3.04516347e-03,
                7.45504877e01,
            ]
        ]
    )

    R = np.array(
        [
            [0.97743098, 0.00689964, 0.21114231],
            [-0.00564446, 0.99996264, -0.00654684],
            [-0.2111796, 0.0052073, 0.97743341],
        ]
    )

    T = np.array([[-476.53571438], [4.78988367], [49.50495583]])
    sensor_width_mm = 6.413  # 传感器宽度，以毫米为单位
    image_width_pixels = 3840  # 图像宽度，以像素为单位
    # 计算每个像素的宽度（以毫米为单位）
    pixel_width_mm = sensor_width_mm / image_width_pixels

    # 创建ImageUndistorter对象
    dual_stereo_calibration = DualStereoCalibration(
        camera_matrix_left,
        camera_matrix_right,
        distortion_coefficients_left,
        distortion_coefficients_right,
        R,
        T,
        pixel_width_mm,
    )

    # # 输入图像路径
    left_image_path = "../assets/template/2circles/2circles-6_5-3-500pixel.png"
    left_image = cv2.imread(left_image_path)
    right_image_path = "../assets/template/2circles/2circles-6_5-3-500pixel.png"
    right_image = cv2.imread(right_image_path)
    print(left_image.shape)
    print(right_image.shape)

    # # 输出图像路径
    left_output_path = "../results/2circles-6_5-3-500pixel_left.jpg"
    right_output_path1 = "../results/2circles-6_5-3-500pixel_right.jpg"

    # # 调用函数进行图像畸变矫正
    (
        rectified_image_left,
        rectified_image_right,
        undistorted_left,
        undistorted_right,
        R1,
        R2,
        P1,
        P2,
        Q,
        roi1,
        roi2,
    ) = dual_stereo_calibration.undistort_images(left_image, right_image)

    cv2.imwrite(left_output_path, rectified_image_left)
    cv2.imwrite(right_output_path1, rectified_image_right)

    # 这里进行模板匹配中心点识别
    # 输入需要矫正的点坐标
    xy = [200.0, 200.0]
    # left 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_point(
        xy, camera_matrix_left, R1, P1
    )
    print(f"left 矫正后的点坐标为: {xy_corrected}")

    # right 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_point(
        xy, camera_matrix_right, R2, P2
    )
    print(f"right 矫正后的点坐标为: {xy_corrected}")

    xy = [[200.0, 200.0]]
    # 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_point(
        xy, camera_matrix_left, R1, P1
    )
    print(f"left 矫正后的点坐标为: {xy_corrected}")

    xy = [[[200.0, 200.0]]]
    # 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_point(
        xy, camera_matrix_left, R1, P1
    )
    print(f"left 矫正后的点坐标为: {xy_corrected}")

    xy = [[200.0, 200.0], [400.0, 400.0]]
    # 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_points(
        xy, camera_matrix_left, R1, P1
    )
    print(f"left 矫正后的点坐标为: {xy_corrected}")

    print("=" * 100)
    # 测试为None的情况
    dual_stereo_calibration = DualStereoCalibration()
    dual_stereo_calibration.undistort_images(left_image, right_image)
    xy = [200.0, 200.0]
    xy_corrected = dual_stereo_calibration.undistort_point(xy, None, None, None)
    print(f"矫正后的点坐标为: {xy_corrected}")
    xy = [[200.0, 200.0], [400.0, 400.0]]
    xy_corrected = dual_stereo_calibration.undistort_points(xy, None, None, None)
    print(f"矫正后的点坐标为: {xy_corrected}")
