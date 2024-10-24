import cv2
import numpy as np
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
        image_size: tuple[int, int] | None = None,  # w, h
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
        self.pixel_width_mm = pixel_width_mm
        self.image_size = image_size  # w, h

        # 获取立体校正参数
        if any(
            [
                self.camera_matrix_left is None,
                self.camera_matrix_right is None,
                self.R is None,
                self.T is None,
                self.image_size is None,
            ]
        ):
            self.R1 = None
            self.R2 = None
            self.P1 = None
            self.P2 = None
            self.Q = None
            self.roi1 = None
            self.roi2 = None
            self.map1_left = None
            self.map2_left = None
            self.map1_right = None
            self.map2_right = None
        else:
            self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = (
                cv2.stereoRectify(
                    self.camera_matrix_left,
                    None,
                    self.camera_matrix_right,
                    None,
                    image_size,
                    self.R,
                    self.T,
                )
            )
            self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
                self.camera_matrix_left,
                None,
                self.R1,
                self.P1,
                image_size,
                cv2.CV_32FC1,
            )
            self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
                self.camera_matrix_right,
                None,
                self.R2,
                self.P2,
                image_size,
                cv2.CV_32FC1,
            )

    def undistort_image(
        self, image: np.ndarray, camera: str = "left"
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """图像畸变矫正"""
        if any(
            [
                self.camera_matrix_left is None,
                self.camera_matrix_right is None,
                self.distortion_coefficients_left is None,
                self.distortion_coefficients_right is None,
                self.map1_left is None,
                self.map2_left is None,
                self.map1_right is None,
                self.map2_right is None,
            ]
        ):
            logger.warning("camera matrices not set, couldn't undistort images.")
            return image, image

        logger.debug(f"undistorting image with {camera} camera")
        if camera == "left":
            # 使用 cv2.undistort 进行畸变矫正
            undistorted_image: np.ndarray = cv2.undistort(
                image, self.camera_matrix_left, self.distortion_coefficients_left
            )
            # 应用校正映射
            rectified_image: np.ndarray = cv2.remap(
                undistorted_image, self.map1_left, self.map2_left, cv2.INTER_LINEAR
            )
        else:
            undistorted_image: np.ndarray = cv2.undistort(
                image, self.camera_matrix_right, self.distortion_coefficients_right
            )
            rectified_image: np.ndarray = cv2.remap(
                undistorted_image, self.map1_right, self.map2_right, cv2.INTER_LINEAR
            )

        return undistorted_image, rectified_image

    def undistort_point(
        self, point: np.ndarray | list[float], camera: str = "left"
    ) -> list[float]:
        """对单个点坐标进行畸变矫正

        Args:
            point (np.ndarray | list[float]): 需要矫正的坐标, 格式为 [x, y]

        Returns:
            list[float]: 矫正后的坐标, 格式为 [x, y]
        """

        # 将点转换为符合输入格式的数组
        point = np.array(point)

        if any(
            [
                self.camera_matrix_left is None,
                self.camera_matrix_right is None,
                self.R1 is None,
                self.R2 is None,
                self.P1 is None,
                self.P2 is None,
            ]
        ):
            logger.warning("camera matrices not set, couldn't undistort points.")
            return point.tolist()

        logger.debug(f"undistorting point with {camera} camera")
        if camera == "left":
            undistorted_point: np.ndarray = cv2.undistortPoints(
                point, self.camera_matrix_left, None, R=self.R1, P=self.P1
            )
        else:
            undistorted_point: np.ndarray = cv2.undistortPoints(
                point, self.camera_matrix_right, None, R=self.R2, P=self.P2
            )
        # 返回矫正后的点坐标
        point_corrected: list = undistorted_point.ravel().tolist()
        return point_corrected

    def undistort_points(
        self, points: np.ndarray | list[list[float]], camera: str = "left"
    ) -> list[list[float]]:
        """对多个点坐标进行畸变矫正

        Args:
            points (np.ndarray | list[list[float]]): 多个坐标, 格式为 [[x1, y1], [x2, y2], ...]

        Returns:
            list[list[float]]: 矫正后的坐标, 格式为 [[x1, y1], [x2, y2], ...]
        """
        points = np.array(points).reshape(-1, 2)

        if any(
            [
                self.camera_matrix_left is None,
                self.camera_matrix_right is None,
                self.R1 is None,
                self.R2 is None,
                self.P1 is None,
                self.P2 is None,
            ]
        ):
            logger.warning("camera matrices not set, couldn't undistort points.")
            return points.tolist()

        logger.debug(f"undistorting points with {camera} camera")
        undistorted_points = []
        for point in points:
            if camera == "left":
                undistorted_point: np.ndarray = cv2.undistortPoints(
                    point, self.camera_matrix_left, None, R=self.R1, P=self.P1
                )
            else:
                undistorted_point: np.ndarray = cv2.undistortPoints(
                    point, self.camera_matrix_right, None, R=self.R2, P=self.P2
                )
            point_corrected: list = undistorted_point.ravel().tolist()
            undistorted_points.append(point_corrected)
        return undistorted_points

    def get_stereo_parameters(
        self,
    ) -> tuple[None, None, None] | tuple[float, float, float]:
        if self.P1 is None or self.P2 is None:
            return None, None, None

        P1, P2 = self.P1, self.P2
        focal_length_pixels = P1[0, 0]
        baseline = -P2[0, 3] / P2[0, 0]
        pixel_width_mm = self.pixel_width_mm  # 使用 self.pixel_width_mm
        focal_length_mm = focal_length_pixels * pixel_width_mm
        return focal_length_pixels, baseline, focal_length_mm

    def pixel_to_world(
        self,
        left_points: np.ndarray | list[list[float]] | list[float],
        right_points: np.ndarray | list[list[float]] | list[float],
    ) -> tuple[float, float, list[float], float]:
        focal_length, baseline, focal_length_mm = self.get_stereo_parameters()
        if focal_length is None or baseline is None or focal_length_mm is None:
            logger.warning("stereo parameters not set, couldn't calculate depth.")
            return 0., 0., [], 0.

        logger.debug("calculating depth")
        left_points = np.array(left_points).reshape(-1, 2)
        right_points = np.array(right_points).reshape(-1, 2)

        points4D: np.ndarray = cv2.triangulatePoints(
            self.P1, self.P2, left_points.T, right_points.T
        )
        points3D: np.ndarray = points4D[:3] / points4D[3]

        disparities = []
        depths = []
        for i in range(points3D.shape[1]):
            left_point = left_points[i]
            right_point = right_points[i]
            disparity = left_point[0] - right_point[0]

            if disparity == 0:
                continue

            disparities.append(disparity)
            depth = (focal_length * baseline) / disparity
            depths.append(depth)

        avg_disparity = float(np.mean(disparities)) if disparities else 0.
        avg_depth = float(np.mean(depths)) if depths else 0.

        return avg_disparity, avg_depth, depths, focal_length_mm


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

    image_size = (500, 500)

    # 创建ImageUndistorter对象
    dual_stereo_calibration = DualStereoCalibration(
        camera_matrix_left,
        camera_matrix_right,
        distortion_coefficients_left,
        distortion_coefficients_right,
        R,
        T,
        pixel_width_mm,
        image_size,
    )

    # # 输入图像路径
    left_image_path = "../assets/template/2circles/2circles-6_5-3-500pixel.png"
    left_image = cv2.imread(left_image_path)
    right_image_path = "../assets/template/2circles/2circles-6_5-3-500pixel.png"
    right_image = cv2.imread(right_image_path)
    print(left_image.shape)
    print(right_image.shape)

    # # 输出图像路径
    left_output_path = "../results/2circles-6_5-3-500pixel_left1.jpg"
    right_output_path1 = "../results/2circles-6_5-3-500pixel_right1.jpg"

    # 调用函数进行图像畸变矫正
    left_undistorted_image, left_rectified_image = (
        dual_stereo_calibration.undistort_image(left_image, "left")
    )
    right_undistorted_image, right_rectified_image = (
        dual_stereo_calibration.undistort_image(right_image, "right")
    )

    cv2.imwrite(left_output_path, left_undistorted_image)
    cv2.imwrite(right_output_path1, right_undistorted_image)

    # 这里进行模板匹配中心点识别
    # 输入需要矫正的点坐标
    xy = [200.0, 200.0]
    # left 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_point(xy, "left")
    print(f"left 矫正后的点坐标为: {xy_corrected}")

    # right 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_point(xy, "right")
    print(f"right 矫正后的点坐标为: {xy_corrected}")

    xy = [[200.0, 200.0]]
    # 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_point(xy, "left")
    print(f"left 矫正后的点坐标为: {xy_corrected}")

    xy = [[[200.0, 200.0]]]
    # 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_point(xy, "left")
    print(f"left 矫正后的点坐标为: {xy_corrected}")

    xy = [[200.0, 200.0], [400.0, 400.0]]
    # 畸变矫正后的坐标点
    xy_corrected = dual_stereo_calibration.undistort_points(xy, "left")
    print(f"left 矫正后的点坐标为: {xy_corrected}")

    left_point = [200, 200]
    right_point = [220, 200]
    avg_disparity, avg_depth, depths, focal_length_mm = (
        dual_stereo_calibration.pixel_to_world(left_point, right_point)
    )
    print("Average Disparity:", avg_disparity)
    print("Average Depth:", avg_depth)

    left_points = [[200, 200], [300, 300]]
    right_points = [[220, 200], [320, 300]]
    avg_disparity, avg_depth, depths, focal_length_mm = (
        dual_stereo_calibration.pixel_to_world(left_points, right_points)
    )
    print("Average Disparity:", avg_disparity)
    print("Average Depth:", avg_depth)

    print("=" * 100)
    # 测试为None的情况
    dual_stereo_calibration = DualStereoCalibration()
    dual_stereo_calibration.undistort_image(left_image)
    xy = [200.0, 200.0]
    xy_corrected = dual_stereo_calibration.undistort_point(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")
    xy = [[200.0, 200.0], [400.0, 400.0]]
    xy_corrected = dual_stereo_calibration.undistort_points(xy)
    print(f"矫正后的点坐标为: {xy_corrected}")

    left_point = [200, 200]
    right_point = [220, 200]
    avg_disparity, avg_depth, depths, focal_length_mm = (
        dual_stereo_calibration.pixel_to_world(left_point, right_point)
    )
    print("Average Disparity:", avg_disparity)
    print("Average Depth:", avg_depth)

    left_points = [[200, 200], [300, 300]]
    right_points = [[220, 200], [320, 300]]
    avg_disparity, avg_depth, depths, focal_length_mm = (
        dual_stereo_calibration.pixel_to_world(left_points, right_points)
    )
    print("Average Disparity:", avg_disparity)
    print("Average Depth:", avg_depth)
