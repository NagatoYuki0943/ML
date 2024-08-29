from threading import Lock
from typing import Literal
from pathlib import Path
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
    """主进程配置
    """
    lock = Lock()   # 锁, 在读取或者修改配置文件时要加锁
    main_sleep_interval: int = 500  # 主循环 sleep_time ms
    save_dir: Path = Path("results")


@dataclass
class CameraConfig(BaseConfig):
    """相机配置
    """
    lock = Lock()
    exposure_time: int = 40000                              # 曝光时间 微秒
    analogue_gain: float = None                             # 模拟增益
    capture_time_interval: int = 1000                       # 相机拍照间隔 ms
    return_image_time_interval: int = 5000                  # 返回图片的检测 ms
    capture_mode: Literal['preview', 'low', 'full'] = 'full'# 相机拍照模式
    queue_maxsize: int = 5                                  # 相机拍照队列最大长度
    camera_left_index: int = 1                              # 左侧相机 index
    camera_right_index: int = 0                             # 右侧相机 index


@dataclass
class MatchTemplateConfig(BaseConfig):
    """模板匹配配置
    """
    lock = Lock()
    template_path: Path = Path("assets/template/circles2-7.5cm-390.png")
    match_method: int = cv2.TM_CCOEFF_NORMED
    init_scale: float = 0.075   # 8 mm: 0.025, 12 mm: 0.03, 25 mm: 0.075, 35 mm: 0.085, 50 mm: 0.15, 15m: 0.01
    scales: tuple[float] = (1.0, 4.0, 0.1)
    target_number: int = 2
    iou_threshold: float = 0.5
    use_threshold_match: bool = True
    threshold_match_threshold: float = 0.6
    threshold_iou_threshold: float = 0.5


@dataclass
class AdjustCameraConfig(BaseConfig):
    """调整相机配置
    """
    lock = Lock()
    mean_light_suitable_range: tuple[float] = (100, 160)
    adjust_exposure_time_step: int = 1000
    capture_mode: Literal['preview', 'low', 'full'] = 'low'
    capture_time_interval: int = 100
    return_image_time_interval: int = 300


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
