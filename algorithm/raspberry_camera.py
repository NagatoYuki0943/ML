import numpy as np
import time
from pathlib import Path
from datetime import datetime
from typing import Literal
from loguru import logger
from picamera2 import Picamera2, Preview
from picamera2.request import CompletedRequest


class RaspberryCameras:
    def __init__(
        self,
        camera_indexes: int | list[int] | tuple[int] = 0,
        log_level: int = "INFO",
        log_file_path: str | Path = Path("logs/camera.log"),
        low_res_ratio: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            camera_indexes (int | list[int] | tuple[int], optional): 相机 index. Defaults to 0.
            log_level (str, optional): log 级别. Defaults to 'INFO'.
            log_file_path (str | Path, optional): 日志文件路径. Defaults to "logs/camera.log".
            low_res_ratio (float, optional): 低分辨率比率. Defaults to 0.5.
        """
        # 日志
        log_file_path = Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_file_path, "a", encoding="utf-8")
        _log_level = {
            "TRACE": Picamera2.DEBUG,
            "DEBUG": Picamera2.DEBUG,
            "INFO": Picamera2.INFO,
            "SUCCESS": Picamera2.INFO,
            "WARNING": Picamera2.WARNING,
            "ERROR": Picamera2.ERROR,
            "CRITICAL": Picamera2.CRITICAL,
        }[log_level]
        Picamera2.set_logging(_log_level, self.log_file)

        camera_indexes = (
            [camera_indexes] if isinstance(camera_indexes, int) else camera_indexes
        )
        self.camera_indexes = camera_indexes

        # 相机
        self.picam2s = {}
        # 预览分辨率
        self.preview_configs = {}
        # 低分辨率 configs
        self.low_res_configs = {}
        # 高分辨率 configs
        self.full_res_configs = {}
        for camera_index in camera_indexes:
            picam2 = Picamera2(camera_index)
            # 相机
            self.picam2s[camera_index] = picam2

            # 预览分辨率
            self.preview_configs[camera_index] = picam2.create_preview_configuration(
                main={"format": "BGR888"}
            )

            # 低分辨率
            sensor_resolution = picam2.sensor_resolution
            self.low_res_configs[camera_index] = picam2.create_still_configuration(
                main={
                    "format": "BGR888",  # 默认就是 BGR888, RGB格式
                    "size": (
                        int(sensor_resolution[0] * low_res_ratio),
                        int(sensor_resolution[1] * low_res_ratio),
                    ),
                }
            )

            # 全分辨率, 默认就是 BGR888, RGB格式
            full_res_config = picam2.create_still_configuration()
            self.full_res_configs[camera_index] = full_res_config
            # 默认设置为高分辨率 config
            picam2.configure(full_res_config)

    def close_log_file(self) -> None:
        """关闭日志文件"""
        self.log_file.close()

    def start_preview(
        self,
        camera_indexes: int | list[int] | tuple[int] = 0,
        preivew: bool | Preview = True,
    ) -> None:
        """开启预览模式

        Args:
            camera_indexes (int | list[int] | tuple[int], optional): 相机 index. Defaults to 0.
            preivew (bool | Preview, optional): 预览模式. Defaults to True.
                preview 设置为 True, 自动判断显示状态
                NULL：值为 0，表示不使用任何预览窗口。这通常用于后台处理图像数据，而不需要在屏幕上显示预览。
                DRM：值为 1，代表 Direct Rendering Manager。适用于 Non-GUI 环境。
                QT：值为 2，代表使用 Qt 框架进行预览。。
                QTGL：值为 3，代表使用 Qt 和 OpenGL 进行预览。结合 Qt 使用 OpenGL 可以提供更丰富的图形效果和更高效的渲染性能。
        """
        camera_indexes = (
            [camera_indexes] if isinstance(camera_indexes, int) else camera_indexes
        )
        for camera_index in camera_indexes:
            self.picam2s[camera_index].start_preview(preivew)
            logger.success(f"camera {camera_index} preview started!")

    def start_preview_all(
        self,
        preivew: bool | Preview = True,
    ) -> None:
        """开启预览模式

        Args:
            preivew (bool | Preview, optional): 预览模式. Defaults to True.
                preview 设置为 True, 自动判断显示状态
                NULL：值为 0，表示不使用任何预览窗口。这通常用于后台处理图像数据，而不需要在屏幕上显示预览。
                DRM：值为 1，代表 Direct Rendering Manager。适用于 Non-GUI 环境。
                QT：值为 2，代表使用 Qt 框架进行预览。。
                QTGL：值为 3，代表使用 Qt 和 OpenGL 进行预览。结合 Qt 使用 OpenGL 可以提供更丰富的图形效果和更高效的渲染性能。
        """
        for picam2 in self.picam2s.values():
            picam2.start_preview(preivew)
        logger.success("all camera preview started!")

    def start(
        self,
        camera_indexes: int | list[int] | tuple[int] = 0,
    ) -> None:
        """开启相机

        Args:
            camera_indexes (int | list[int] | tuple[int], optional): 相机 index. Defaults to 0.
        """
        camera_indexes = (
            [camera_indexes] if isinstance(camera_indexes, int) else camera_indexes
        )
        for camera_index in camera_indexes:
            self.picam2s[camera_index].start()
            logger.success(f"camera {camera_index} started!")

    def start_all(self) -> None:
        """开启所有相机"""
        for picam2 in self.picam2s.values():
            picam2.start()
        logger.success("all cameras started!")

    def stop(
        self,
        camera_indexes: int | list[int] | tuple[int] = 0,
    ) -> None:
        """停止相机

        Args:
            camera_indexes (int | list[int] | tuple[int], optional): 相机 index. Defaults to 0.
        """
        camera_indexes = (
            [camera_indexes] if isinstance(camera_indexes, int) else camera_indexes
        )
        for camera_index in camera_indexes:
            self.picam2s[camera_index].stop()
            logger.success(f"camera {camera_index} stoped!")

    def stop_all(self) -> None:
        """停止所有相机"""
        for picam2 in self.picam2s.values():
            picam2.stop()
        logger.success("all cameras stoped!")

    def close(
        self,
        camera_indexes: int | list[int] | tuple[int] = 0,
    ) -> None:
        """关闭相机

        Args:
            camera_indexes (int | list[int] | tuple[int], optional): 相机 index. Defaults to 0.
        """
        camera_indexes = (
            [camera_indexes] if isinstance(camera_indexes, int) else camera_indexes
        )
        for camera_index in camera_indexes:
            self.picam2s[camera_index].close()
            logger.success(f"camera {camera_index} closeed!")

    def close_all(self) -> None:
        """关闭所有相机"""
        for picam2 in self.picam2s.values():
            picam2.close()
        logger.success("all cameras closed!")

    def switch_mode(
        self,
        camera_indexes: int | list[int] | tuple[int] = 0,
        capture_mode: Literal["preview", "low", "full"] = "full",
    ) -> None:
        """切换拍照模式

        Args:
            camera_indexes (int | list[int] | tuple[int], optional): 相机 index. Defaults to 0.
            capture_mode (Literal['full', 'low'], optional): 相机模式. Defaults to 'full'.
        """
        assert capture_mode in [
            "preview",
            "low",
            "full",
        ], f"capture_mode must in ['preview', 'low', 'full'], but got {capture_mode}."

        camera_indexes = (
            [camera_indexes] if isinstance(camera_indexes, int) else camera_indexes
        )
        for camera_index in camera_indexes:
            # 获取拍照配置
            if capture_mode == "preview":
                camera_config = self.preview_configs[camera_index]
            elif capture_mode == "low":
                camera_config = self.low_res_configs[camera_index]
            else:
                camera_config = self.full_res_configs[camera_index]

            self.picam2s[camera_index].switch_mode(camera_config)
            logger.success(f"camera {camera_index} switch mode to {capture_mode}.")

    def switch_mode_all(
        self,
        capture_mode: Literal["preview", "low", "full"] = "full",
    ) -> None:
        """切换所有相机拍照模式

        Args:
            capture_mode (Literal['preview','full', 'low'], optional): 相机模式. Defaults to 'full'.
        """
        assert capture_mode in [
            "preview",
            "low",
            "full",
        ], f"capture_mode must in ['preview', 'low', 'full'], but got {capture_mode}."
        # 获取拍照配置
        for camera_index, picam2 in self.picam2s.items():
            # 获取拍照配置
            if capture_mode == "preview":
                camera_config = self.preview_configs[camera_index]
            elif capture_mode == "low":
                camera_config = self.low_res_configs[camera_index]
            else:
                camera_config = self.full_res_configs[camera_index]
            picam2.switch_mode(camera_config)
        logger.success(f"all camera switch mode to {capture_mode}.")

    def capture(
        self,
        camera_index: int | None = None,
        ExposureTime: int | None = None,
        AnalogueGain: float | None = None,
        timestamp: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """拍照

        Args:
            camera_index (int | None, optional): 相机 index . Defaults to None, 代表使用 indexes 中的第一个相机.
            ExposureTime (int | None, optional): 曝光时间，单位微秒. Defaults to None.
            AnalogueGain (float | None, optional): 模拟增益. Defaults to None.
            timestamp (str | None, optional): 时间戳. Defaults to None.

        Returns:
            tuple[np.ndarray, dict]: 拍摄的照片和拍摄的 metadata
        """
        timestamp = (
            datetime.now().strftime("%Y%m%d-%H%M%S.%f")
            if timestamp is None
            else timestamp
        )

        camera_index = self.camera_indexes[0] if camera_index is None else camera_index
        picam2: Picamera2 = self.picam2s[camera_index]
        # 在这种情况下，我们注意到with构造的使用。虽然您通常可以不用它（只需直接设置picam2.controls），但这并不能绝对保证这两个控件都应用于同一帧。
        with picam2.controls as controls:
            if ExposureTime is not None:
                controls.ExposureTime = ExposureTime
            if AnalogueGain is not None:
                controls.AnalogueGain = AnalogueGain

        request: CompletedRequest = picam2.capture_request()
        array: np.ndarray = request.make_array("main")
        metadata = request.get_metadata()  # this is the metadata for this image
        request.release()
        # logger.info(
        #     f"{timestamp} ExposureTime = {metadata['ExposureTime']}, AnalogueGain = {metadata['AnalogueGain']}"
        # )

        return array, metadata


def test_raspberry_cameras_single() -> None:
    import cv2

    camera_index = 0
    # 初始化类
    raspberry_cameras = RaspberryCameras(camera_index)

    # 测试预览，可以注释掉
    # raspberry_cameras.start_preview(camera_index, preivew=True)
    # raspberry_cameras.start_preview_all(preivew=True)

    # 启动相机
    raspberry_cameras.start(camera_index)
    time.sleep(1)

    # 拍摄图片
    image, metadata = raspberry_cameras.capture(camera_index)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("single-full-resolution-1.jpg", image)

    # 切换拍照模式
    raspberry_cameras.switch_mode(camera_index, "preview")
    # 拍摄图片
    image, metadata = raspberry_cameras.capture(camera_index)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("single-preview.jpg", image)

    # 切换拍照模式
    raspberry_cameras.switch_mode(camera_index, "low")
    # 拍摄图片
    image, metadata = raspberry_cameras.capture(camera_index)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("single-low-resolution.jpg", image)

    # 切换拍照模式
    raspberry_cameras.switch_mode(camera_index, "full")
    # 拍摄图片
    image, metadata = raspberry_cameras.capture(camera_index)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("single-full-resolution-2.jpg", image)

    # 停止相机
    raspberry_cameras.stop(camera_index)

    # 关闭相机
    raspberry_cameras.close(camera_index)

    # 关闭日志文件
    raspberry_cameras.close_log_file()


def test_raspberry_cameras_double() -> None:
    import cv2

    camera_indexes = [0, 1]
    # 初始化类
    raspberry_cameras = RaspberryCameras(camera_indexes)

    # 启动相机
    raspberry_cameras.start_all()
    time.sleep(1)

    for camera_index in camera_indexes:
        # 拍摄图片
        image, metadata = raspberry_cameras.capture(camera_index)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"double-full-resolution-{camera_index}.jpg", image)

    # 停止相机
    raspberry_cameras.stop_all()

    # 关闭相机
    raspberry_cameras.close_all()

    # 关闭日志文件
    raspberry_cameras.close_log_file()


def test_raspberry_cameras_speed() -> None:
    camera_index = 0
    # 初始化类
    raspberry_cameras = RaspberryCameras(camera_index)

    # 启动相机
    raspberry_cameras.start(camera_index)
    time.sleep(1)

    raspberry_cameras.switch_mode_all("low")

    capture_times = 100
    time_sum = 0
    for i in range(capture_times):
        begin = time.time()
        # 拍摄图片
        image, metadata = raspberry_cameras.capture(camera_index)
        end = time.time()
        time_interval = end - begin
        logger.info(f"capture {i} time interval = {time_interval}")
        time_sum += time_interval

    logger.info(f"mean time interval = {time_sum / capture_times}")
    # full: mean time interval = 0.2058906626701355
    # low:  mean time interval = 0.05599453449249268

    # 停止相机
    raspberry_cameras.stop(camera_index)

    # 关闭相机
    raspberry_cameras.close(camera_index)

    # 关闭日志文件
    raspberry_cameras.close_log_file()


def test_restart_raspberry_cameras() -> None:
    camera_index = 0

    print("start 1")
    raspberry_cameras = RaspberryCameras(camera_index)
    raspberry_cameras.start(camera_index)
    raspberry_cameras.capture(camera_index)
    raspberry_cameras.close(camera_index)
    raspberry_cameras.close_log_file()
    id1 = id(raspberry_cameras)
    print("end 1")

    print("start 2")
    raspberry_cameras = RaspberryCameras(camera_index)
    raspberry_cameras.start(camera_index)
    raspberry_cameras.capture(camera_index)
    raspberry_cameras.close(camera_index)
    raspberry_cameras.close_log_file()
    id2 = id(raspberry_cameras)
    print("end 2")
    print(id1)
    print(id2)
    print(id1 == id2)


if __name__ == "__main__":
    test_raspberry_cameras_single()
    # test_raspberry_cameras_double()
    test_raspberry_cameras_speed()
    test_restart_raspberry_cameras()
