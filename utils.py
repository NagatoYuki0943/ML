from queue import Queue
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from config import MainConfig


def clear_queue(*queues: Queue):
    queue: Queue
    for queue in queues:
        try:
            # 使用 timeout 参数防止无限等待
            while not queue.empty():
                # 设置超时时间为 1 秒以避免死锁
                queue.get(timeout=1)
        except queue.Empty:
            # 当队列为空时继续处理下一个队列
            continue
        except Exception as e:
            # 处理其他可能的异常
            print(f"An error occurred: {e}")


def drop_excessive_queue_items(queue: Queue):
    camera_qsize = queue.qsize()
    # 忽略多余的图片
    if camera_qsize > 1:
        logger.warning(f"queue got {camera_qsize} items, ignore {camera_qsize - 1} itmes")
        for _ in range(camera_qsize - 1):
            try:
                queue.get(timeout=MainConfig.getattr("get_picture_timeout"))
            except queue.Empty:
                logger.error("get item timeout")


def enhance_contrast_clahe(image: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """clahe增强对比度"""
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 应用CLAHE
    enhanced_image = clahe.apply(image)

    return enhanced_image


def save_to_jsonl(data, file_path: str | Path, mode: str = 'a'):
    """
    Save data to a JSON lines file.
    """
    if data is None:
        # write an empty file if data is None
        with open(file_path, mode='w', encoding='utf-8') as f:
            f.write('')
        return
    string = json.dumps(data, ensure_ascii=False)
    with open(file_path, mode=mode, encoding='utf-8') as f:
        f.write(string + "\n")


def load_standard_cycle_results(file_path: str | Path) -> dict | None:
    file_path = Path(file_path)
    if file_path.exists():
        with open(file_path, mode='r', encoding='utf-8') as f:
            # 读取第一行数据
            line = f.readline()
            if line:
                data = json.loads(line)
                if data is not None:
                    data = {eval(k): v for k, v in data.items()}
                    return data


def test_load_standard_cycle_results():
    data = load_standard_cycle_results('results/history.jsonl')
    print(data)


def get_now_time():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')


def save_image(image: np.ndarray, file_path: str | Path):
    image = image.copy()
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(object=file_path), image)


def get_picture_timeout_process():
    logger.error("get picture timeout")
    MainConfig.setattr(
        "get_picture_timeout_count",
        MainConfig.getattr("get_picture_timeout_count") + 1
    )


if __name__ == '__main__':
    test_load_standard_cycle_results()
