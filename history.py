import numpy as np
from pathlib import Path
import yaml
from copy import deepcopy


def save_history(history: dict, path: str | Path = 'history.yaml') -> None:
    history = deepcopy(history)
    for key, value in history.items():
        for k, v in value.items():
            if isinstance(v, np.ndarray):
                history[key][k] = v.tolist()

    with open(path, 'w') as f:
        yaml.dump(history, f)


def load_history(path: str | Path = 'history.yaml') -> dict:
    with open(path, 'r') as f:
        history = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in history.items():
        for k, v in value.items():
            if isinstance(v, list):
                history[key][k] = np.array(v)
    return history


def test_history():
    history = {
        0: {'image_timestamp': '20240907-114158.188148', 'box': np.array([1671,  935, 1990, 1254]), 'center': np.array([1830.76552951, 1094.84461901]), 'exposure_time': 84000},
        1: {'image_timestamp': '20240907-114158.188148', 'box': np.array([1840, 1895, 2341, 2396]), 'center': np.array([2090.62638041, 2146.0040168 ]), 'exposure_time': 84000}
    }
    save_history(history)
    loaded_history = load_history()
    print(loaded_history)
    {
        0: {'box': np.array([1671,  935, 1990, 1254]), 'center': np.array([1830.76552951, 1094.84461901]), 'exposure_time': 84000, 'image_timestamp': '20240907-114158.188148'},
        1: {'box': np.array([1840, 1895, 2341, 2396]), 'center': np.array([2090.62638041, 2146.0040168 ]), 'exposure_time': 84000, 'image_timestamp': '20240907-114158.188148'}
    }


if __name__ == '__main__':
    test_history()
