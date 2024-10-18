from loguru import logger
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


class DualStereoCalibration:
    def __init__(
        self,
        camera_matrix_left: np.ndarray | list[list[float]],
        camera_matrix_right: np.ndarray | list[list[float]],
        distortion_coefficients_left: np.ndarray | list[list[float]],
        distortion_coefficients_right: np.ndarray | list[list[float]],
        R: np.ndarray | list[list[float]],
        T: np.ndarray | list[list[float]],
        pixel_width_mm: float,
    ):
        self.camera_matrix_left = np.array(camera_matrix_left)
        self.camera_matrix_right = np.array(camera_matrix_right)
        self.distortion_coefficients_left = np.array(distortion_coefficients_left)
        self.distortion_coefficients_right = np.array(distortion_coefficients_right)
        self.R = np.array(R)
        self.T = np.array(T)
        self.pixel_width_mm = pixel_width_mm

    def rectify_images(self, image_left: np.ndarray, image_right: np.ndarray):
        image_size = (image_left.shape[1], image_left.shape[0])

        # 使用 cv2.undistort 进行畸变矫正
        undistorted_left = cv2.undistort(
            image_left, self.camera_matrix_left, self.distortion_coefficients_left
        )
        undistorted_right = cv2.undistort(
            image_right, self.camera_matrix_right, self.distortion_coefficients_right
        )
        focal_length_pixels_R = self.camera_matrix_right[0, 0]
        focal_length_pixels_L = self.camera_matrix_left[0, 0]
        # logger.info(f"Focal length (in pixels): {focal_length_pixels}")
        # 给定的传感器尺寸和图像分辨率
        # sensor_width_mm = 6.413  # 传感器宽度，以毫米为单位
        # image_width_pixels = 3840  # 图像宽度，以像素为单位

        # 计算每个像素的宽度（以毫米为单位）
        # pixel_width_mm = sensor_width_mm / image_width_pixels

        # 将焦距从像素转换为毫米
        focal_length_mm_R = focal_length_pixels_R * self.pixel_width_mm
        logger.info("R origin focal", focal_length_mm_R)
        focal_length_mm_L = focal_length_pixels_L * self.pixel_width_mm
        logger.info("R origin focal", focal_length_mm_L)
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
            R1,
            R2,
            P1,
            P2,
            Q,
            roi1,
            roi2,
            undistorted_left,
            undistorted_right,
        )

    def undistort_points(self, points: np.ndarray, camera_matrix, R, P):
        points = points.reshape(-1, 2)
        rectified_points = []
        for point in points:
            new_point = cv2.undistortPoints(
                np.array([[point]]), camera_matrix, None, R=R, P=P
            )
            x, y = new_point.ravel()
            rectified_points.append([x, y])
        return rectified_points

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
    pass
