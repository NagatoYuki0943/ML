from threading import Lock
from typing import Literal
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Any
from loguru import logger
import yaml


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
    log_level: Literal['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
    save_dir: Path = Path("results")
    save_dir.mkdir(parents=True, exist_ok=True)
    location_save_dir = save_dir / "rings_location"
    location_save_dir.mkdir(parents=True, exist_ok=True)
    camera_result_save_path = save_dir / "camera_result.jsonl"
    left_camera_result_save_path = save_dir / "left_result.jsonl"
    right_camera_result_save_path = save_dir / "right_result.jsonl"
    calibration_result_save_path = save_dir / "calibration_result.jsonl"
    history_save_path = save_dir / "history.jsonl"
    standard_save_path = save_dir / "standard.jsonl"
    original_config_path = save_dir / "config_original.yaml"   # 原始 config, 用于重置
    runtime_config_path = save_dir / "config_runtime.yaml" # 运行时 config, 用于临时修改配置
    main_sleep_interval: int = 500  # 主循环 sleep_time ms
    get_picture_timeout: int = 10       # 获取图片超时时间 s
    cycle_time_interval: int = 10000    # 主循环时间 ms
    defalut_error_distance: float = 1e6    # 错误默认距离 m


@dataclass
class CameraConfig(BaseConfig):
    """相机配置
    """
    lock = Lock()
    low_res_ratio: float = 0.5                              # 相机拍摄低分辨率比率
    exposure_time: int = 40000                              # 曝光时间 us
    analogue_gain: float = None                             # 模拟增益
    capture_time_interval: int = 1000                       # 相机拍照间隔 ms
    return_image_time_interval: int = 3000                  # 返回图片的检测 ms
    capture_mode: Literal['preview', 'low', 'full'] = 'full'# 相机拍照模式
    queue_maxsize: int = 5                                  # 相机拍照队列最大长度
    camera_left_index: int = 1                              # 左侧相机 index
    camera_right_index: int = 0                             # 右侧相机 index
    output_format: Literal['rgb', 'gray'] = 'gray'          # 输出格式
    has_filter_plate: bool = True                           # 是否有滤镜板


@dataclass
class AdjustCameraConfig(BaseConfig):
    """调整相机配置
    """
    lock = Lock()
    mean_light_suitable_range: tuple[float] = (70, 160) # (100, 160)
    suitable_ignore_ratio: float = 0.1                 # 忽略 mean_light_suitable_range 最低和最高范围的百分比 [0, 100] -> [10, 90]
    adjust_exposure_time_step: int = 2000
    capture_mode: Literal['preview', 'low', 'full'] = 'low'
    capture_time_interval: int = 100        # 拍照间隔 us
    return_image_time_interval: int = 100   # 返回图片间隔 us
    adjust_total_times: int = 100           # 最高调整次数


@dataclass
class StereoCalibrationConfig(BaseConfig):
    """畸变矫正配置
    """
    lock = Lock()
    camera_matrix_left = [
        [7.44937603e+03, 0.00000000e+00, 1.79056889e+03],
        [0.00000000e+00, 7.45022891e+03, 1.26665786e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]
    camera_matrix_right = [
        [7.46471035e+03, 0.00000000e+00, 1.81985040e+03],
        [0.00000000e+00, 7.46415680e+03, 1.38081032e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]
    distortion_coefficients_left = [[-4.44924086e-01, 6.27814725e-01, -1.80510014e-03, -8.97545764e-04, -1.84473439e+01]]
    distortion_coefficients_right = [[-4.07660445e-01, -2.23391154e+00, -1.09115383e-03, -3.04516347e-03, 7.45504877e+01]]
    R = [
        [0.97743098, 0.00689964, 0.21114231],
        [-0.00564446, 0.99996264, -0.00654684],
        [-0.2111796, 0.0052073, 0.97743341]
    ]
    T = [[-476.53571438], [4.78988367], [49.50495583]]
    # 给定的传感器尺寸和图像分辨率
    sensor_width_mm = 6.413  # 传感器宽度，以毫米为单位
    image_width_pixels = 3840  # 图像宽度，以像素为单位
    # 计算每个像素的宽度（以毫米为单位）
    pixel_width_mm = sensor_width_mm / image_width_pixels


def get_reference_target_ids():
    return [0]


@dataclass
class MatchTemplateConfig(BaseConfig):
    """模板匹配配置
    """
    lock = Lock()
    template_size: tuple[int] = (100, 100)      # 模板大小 (h, w), 单位为 mm
    template_path: Path = Path("assets/template/circles2-7.5cm-390.png")
    match_method: int = cv2.TM_CCOEFF_NORMED    # 匹配方法
    init_scale: float = 0.075                   # 初始 scale 8 mm: 0.025, 12 mm: 0.03, 25 mm: 0.075, 35 mm: 0.085, 50 mm: 0.15, 15m: 0.01
    scales: tuple[float] = (1.0, 4.0, 0.1)      # 缩放 scale 范围 (start, end, step)
    new_target_scales: tuple[float] = (0.5, 1.5, 0.1)  # 新目标的缩放 scale 范围 (start, end, step)
    max_target_number: int = 10                 # 最大目标数量
    target_number: int = 0                      # 默认靶标数量,初始化时为找到的靶标数量
    got_target_number: int = 0                  # 找到的靶标数量
    iou_threshold: float = 0.5                  # iou 阈值
    use_threshold_match: bool = True            # 是否使用阈值匹配
    threshold_match_threshold: float = 0.6      # 阈值匹配阈值
    threshold_iou_threshold: float = 0.5        # 阈值匹配 iou 阈值
    # ratios: np.ndarray = None                 # 模板缩放比率 [...]
    # scores: np.ndarray = None                 # 匹配得分 [...]
    # boxes: np.ndarray = None                  # 匹配的 boxes [[x1, y1, x2, y2], ...]
    # boxes_status: np.ndarray = None           # 当前 box 状态，用 True 代表找得到，False 代表丢失
    id2boxstate: dict[int, dict] | None = None  # 靶标 id 到 boxes 的映射
    reference_target_ids: tuple[int] = tuple()  # 参考靶标 id
    search_range: float = 1                     # 假设为1，box 为 [x1, y1, x2, y2], w, h, 则搜索范围为 [x1 - 1 * w, y1 - 1 * h, x2 + 1 * w, y2 + 1 * h]


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
    move_threshold: float = 0.3     # 定位误差阈值, pixel


@dataclass
class SerialCommConfig(BaseConfig):
    """串口通讯模块配置
    """
    # 串口配置
    port: str ="/dev/ttyAMA2"
    baudrate: int = 115200
    BUFFER_SIZE: int = 2048
    timeout: float = 0
    log_dir: Path = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    temperature_data_save_path = log_dir / "temperature_data.json"
    LOG_SIZE: int = 10_000_000 # 温度数据记录文件大小为10MB


@dataclass
class MQTTConfig(BaseConfig):
    """MQTT客户端配置
    """
    broker: str = "47.116.118.93"
    port: int = 1883
    timeout: int = 60
    topic: str = "$creq/7804d2/+"
    username: str = "admin"
    password: str = "123456"
    clientId: str = "7804d2"
    apikey: str = "123456"
    did: str = "7804d2"


@dataclass
class FTPConfig(BaseConfig):
    """FTP配置
    """
    ip: str = "localhost"
    port: int = 21
    username: str = "admin"
    password: str = "123456"


ALL_CONFIGS = [
    MainConfig,
    CameraConfig,
    AdjustCameraConfig,
    StereoCalibrationConfig,
    MatchTemplateConfig,
    RingsLocationConfig,
    SerialCommConfig,
    MQTTConfig,
    FTPConfig,
]


def save_config_to_yaml(
    configs: list[BaseConfig] = ALL_CONFIGS,
    config_path: str | Path = "config.yaml"
):
    """
    Save a configuration class to a YAML file.

    :param configs: The configuration class to update. If None, all configuration classes will be updated. default: ALL_CONFIGS
    :param config_path: The path to the YAML file to load. default: "config.yaml"
    """
    class2data = {}
    for config in configs:
        data = {}
        for attr in dir(config):
            if not attr.startswith("__") and not callable(getattr(config, attr)) and not attr.startswith("lock"):
                value = config.getattr(attr)
                if isinstance(value, Path):
                    value = str(value)
                data[attr] = value
        class2data[config.__name__] = data

    with open(config_path, 'w') as file:
        yaml.dump(class2data, file, default_flow_style=False)


def load_config_from_yaml(
    configs: list[BaseConfig] = ALL_CONFIGS,
    config_path: str | Path = "config.yaml"
):
    """
    Load configuration from a YAML file and update the given configuration class.

    :param configs: The configuration class to update. If None, all configuration classes will be updated. default: ALL_CONFIGS
    :param config_path: The path to the YAML file to load. default: "config.yaml"
    """
    with open(config_path, 'r') as file:
        class2data = yaml.load(file, Loader=yaml.FullLoader)

    for config in configs:
        data = class2data[config.__name__]
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(getattr(config, key), Path):
                    value = Path(value)
                config.setattr(key, value)


def init_config_from_yaml(
    configs: list[BaseConfig] = ALL_CONFIGS,
    config_path: str | Path = "config.yaml"
):
    """
    初始化配置
    """
    if not Path(config_path).exists():
        save_config_to_yaml(configs, config_path)
    else:
        load_config_from_yaml(configs, config_path)


if __name__ == "__main__":
    save_config_to_yaml(ALL_CONFIGS, MainConfig.original_config_path)
    load_config_from_yaml(ALL_CONFIGS, MainConfig.original_config_path)
    init_config_from_yaml(ALL_CONFIGS, MainConfig.original_config_path)
