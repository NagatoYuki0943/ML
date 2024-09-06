from threading import Lock
from typing import Literal
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Any
from loguru import logger


@dataclass
class BaseConfig:

    @classmethod
    def getattr(cls, attr_name: str) -> Any:
        assert hasattr(cls, attr_name), f"{attr_name} not in {cls.__name__}"
        if hasattr(cls, 'lock'):
            with cls.lock:
                return getattr(cls, attr_name)
        else:
            return getattr(cls, attr_name)

    @classmethod
    def setattr(cls, attr_name: str, value: Any) -> None:
        if hasattr(cls, 'lock'):
            with cls.lock:
                setattr(cls, attr_name, value)
        else:
            setattr(cls, attr_name, value)


@dataclass
class MainConfig(BaseConfig):
    """主线程配置
    """
    lock = Lock()   # 锁, 在读取或者修改配置文件时要加锁
    main_sleep_interval: int = 500  # 主循环 sleep_time ms
    log_level: Literal['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'] = 'DEBUG'
    save_dir: Path = Path("results")
    save_dir.mkdir(parents=True, exist_ok=True)
    location_save_dir = save_dir / "rings_location"
    location_save_dir.mkdir(parents=True, exist_ok=True)
    camera_result_save_path = save_dir / "camera_result.jsonl"
    left_camera_result_save_path = save_dir / "left_result.jsonl"
    right_camera_result_save_path = save_dir / "right_result.jsonl"
    calibration_result_save_path = save_dir / "calibration_result.jsonl"
    get_picture_timeout: int = 10       # 获取图片超时时间 s
    cycle_time_interval: int = 10000    # 主循环时间 ms


@dataclass
class CameraConfig(BaseConfig):
    """相机配置
    """
    lock = Lock()
    low_res_ratio: float = 0.5                              # 相机拍摄低分辨率比率
    exposure_time: int = 40000                              # 曝光时间 us
    analogue_gain: float = None                             # 模拟增益
    capture_time_interval: int = 1000                       # 相机拍照间隔 ms
    return_image_time_interval: int = 2000                  # 返回图片的检测 ms
    capture_mode: Literal['preview', 'low', 'full'] = 'full'# 相机拍照模式
    queue_maxsize: int = 5                                  # 相机拍照队列最大长度
    camera_left_index: int = 1                              # 左侧相机 index
    camera_right_index: int = 0                             # 右侧相机 index


@dataclass
class AdjustCameraConfig(BaseConfig):
    """调整相机配置
    """
    lock = Lock()
    mean_light_suitable_range: tuple[float] = (80, 160) # (100, 160)
    adjust_exposure_time_step: int = 2000
    capture_mode: Literal['preview', 'low', 'full'] = 'low'
    capture_time_interval: int = 100        # 拍照间隔 us
    return_image_time_interval: int = 100   # 返回图片间隔 us
    adjust_total_times: int = 100


@dataclass
class StereoCalibrationConfig(BaseConfig):
    """畸变矫正配置
    """
    lock = Lock()
    camera_matrix_left = np.array([
        [7.44937603e+03, 0.00000000e+00, 1.79056889e+03],
        [0.00000000e+00, 7.45022891e+03, 1.26665786e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    camera_matrix_right = np.array([
        [7.46471035e+03, 0.00000000e+00, 1.81985040e+03],
        [0.00000000e+00, 7.46415680e+03, 1.38081032e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    distortion_coefficients_left = np.array([[-4.44924086e-01, 6.27814725e-01, -1.80510014e-03, -8.97545764e-04, -1.84473439e+01]])
    distortion_coefficients_right = np.array([[-4.07660445e-01, -2.23391154e+00, -1.09115383e-03, -3.04516347e-03, 7.45504877e+01]])
    R = np.array([
        [0.97743098, 0.00689964, 0.21114231],
        [-0.00564446, 0.99996264, -0.00654684],
        [-0.2111796, 0.0052073, 0.97743341]
    ])
    T = np.array([[-476.53571438], [4.78988367], [49.50495583]])
    # 给定的传感器尺寸和图像分辨率
    sensor_width_mm = 6.413  # 传感器宽度，以毫米为单位
    image_width_pixels = 3840  # 图像宽度，以像素为单位
    # 计算每个像素的宽度（以毫米为单位）
    pixel_width_mm = sensor_width_mm / image_width_pixels


@dataclass
class MatchTemplateConfig(BaseConfig):
    """模板匹配配置
    """
    lock = Lock()
    template_path: Path = Path("assets/template/circles2-7.5cm-390.png")
    match_method: int = cv2.TM_CCOEFF_NORMED    # 匹配方法
    init_scale: float = 0.075                   # 初始 scale 8 mm: 0.025, 12 mm: 0.03, 25 mm: 0.075, 35 mm: 0.085, 50 mm: 0.15, 15m: 0.01
    scales: tuple[float] = (1.0, 4.0, 0.1)      # 缩放 scale 范围 (start, end, step)
    target_number: int = 2                      # 靶标数量
    got_target_number: int = 0                  # 找到的靶标数量
    iou_threshold: float = 0.5                  # iou 阈值
    use_threshold_match: bool = True            # 是否使用阈值匹配
    threshold_match_threshold: float = 0.6      # 阈值匹配阈值
    threshold_iou_threshold: float = 0.5        # 阈值匹配 iou 阈值
    ratios: np.ndarray = None                   # 模板缩放比率 [...]
    scores: np.ndarray = None                   # 匹配得分 [...]
    boxes: np.ndarray = None                    # 匹配的 boxes [[x1, y1, x2, y2], ...]
    boxes_status: np.ndarray = None             # 当前 box 状态，用 True 代表找得到，False 代表丢失


@dataclass
class RingsLocationConfig(BaseConfig):
    """圆环定位配置
    """
    lock = Lock()
    gradient_threshold_percent: float = 0.5
    iters: int = 1
    order: int = 2
    rings_nums: int = 6
    min_group_size: int = 5
    sigmas: list[float | int] | float | int = 2
    draw_scale: int = 20
    save_grads: bool = False
    save_detect_images: bool = False
    save_detect_results: bool = False


@dataclass
class SerialCommConfig(BaseConfig):
    """串口通讯模块配置
    """
    lock = Lock()
    # 串口配置
    port: str = "/dev/ttyAMA2"
    baudrate: int = 115200
    BUFFER_SIZE: int = 2048
    timeout: float = 0
    # 温度记录日志配置
    temperature_logger = logger.bind(temperature = True)
    temperature_logger.add(
        "log/temperature_data.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        filter=lambda record: "temperature" in record["extra"],
    )


@dataclass
class MQTTConfig(BaseConfig):
    """MQTT客户端配置
    """
    broker: str = "localhost"
    port: int = 1883
    timeout: int = 60
    topic: str = "test/topic"


@dataclass
class FTPConfig(BaseConfig):
    """FTP配置
    """
    ip: str = ""
    port: int = 0
    username: str = ""
    password: str = ""
