import numpy as np
from algorithm import mean_brightness


def adjust_exposure_by_mean(
    image: np.ndarray,
    exposure_time: float,
    mean_light_suitable_range: tuple[float],
    adjust_exposure_time_step: float = 1000,
) -> tuple[float, str]:
    mean_bright = mean_brightness(image)
    if mean_bright < mean_light_suitable_range[0]:
        return exposure_time + adjust_exposure_time_step, "+"
    elif mean_bright > mean_light_suitable_range[1]:
        return exposure_time - adjust_exposure_time_step, "-"
    else:
        return exposure_time, "ok"
