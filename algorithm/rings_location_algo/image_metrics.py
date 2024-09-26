import numpy as np
from PIL import Image
from scipy import ndimage
import cv2


def mean_brightness(image: np.ndarray | Image.Image) -> np.ndarray:
    image_array = np.asarray(image).astype(np.float64)
    return image_array.mean(axis=(0, 1))


def min_max_contrast(image: np.ndarray | Image.Image) -> np.ndarray:
    image_array = np.asarray(image).astype(np.float64)
    return image_array.max(axis=(0, 1)) - image_array.min(axis=(0, 1))


def get_min_max_contrast_threshold(
    image: np.ndarray | Image.Image, base: float | int = 10
) -> float | int:
    min_max = min_max_contrast(image)
    return min_max / base


def weber_contrast(image: np.ndarray | Image.Image) -> np.ndarray:
    """https://baike.baidu.com/item/%E5%9B%BE%E5%83%8F%E5%AF%B9%E6%AF%94%E5%BA%A6/10850493"""
    raise NotImplementedError


def michelson_contrast(image: np.ndarray | Image.Image) -> np.ndarray:
    """https://baike.baidu.com/item/%E5%9B%BE%E5%83%8F%E5%AF%B9%E6%AF%94%E5%BA%A6/10850493"""
    image_array = np.asarray(image).astype(np.float64)
    _max = image_array.max(axis=(0, 1))
    _min = image_array.min(axis=(0, 1))
    return (_max - _min) / (_max + _min)


def root_mean_square_contrast(image: np.ndarray | Image.Image) -> np.ndarray:
    """https://baike.baidu.com/item/%E5%9B%BE%E5%83%8F%E5%AF%B9%E6%AF%94%E5%BA%A6/10850493"""
    image_array = np.asarray(image).astype(np.float64)
    shape = image_array.shape
    h, w = shape[0], shape[1]
    mean = mean_brightness(image_array)
    mean_square_contrast = np.sum((image_array - mean) ** 2, axis=(0, 1)) / (h * w)
    return np.sqrt(mean_square_contrast)


def image_gradient(
    image: np.ndarray | Image.Image, iters=0
) -> tuple[np.ndarray, np.ndarray]:
    if len(np.shape(image)) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rows, cols = np.shape(image)

    image = np.asarray(image).astype(np.float64)

    image_smooth = image
    # 降噪
    if iters > 0:
        # smooth image
        weight = np.ones((3, 3), dtype=np.float64) / 9
        image_smooth = ndimage.convolve(image, weight, mode="constant")

    # compute partial derivatives 梯度计算
    Gx = np.zeros((rows, cols))
    Gx[0:, 1:-1] = 0.5 * (image_smooth[0:, 2:] - image_smooth[0:, 0:-2])
    Gy = np.zeros((rows, cols))
    Gy[1:-1, 0:] = 0.5 * (image_smooth[2:, 0:] - image_smooth[0:-2, 0:])
    grad = np.sqrt(Gx**2 + Gy**2)  # 计算了每个像素点的梯度幅值

    grad_crop = grad[2:-2, 2:-2]
    return grad, grad_crop


def get_gradient_threshold(gradient: np.ndarray, percent: float = 0.6) -> float:
    return (gradient.max() - gradient.min()) * percent + gradient.min()


def test():
    image = Image.open("squirrel.jpg")
    mean = mean_brightness(image)
    min_max = min_max_contrast(image)
    michelson = michelson_contrast(image)
    rms_contrast = root_mean_square_contrast(image)
    print(f"{mean = }")
    print(f"{min_max = }")
    print(f"{michelson = }")
    print(f"{rms_contrast = }")
    print()

    image_gray = image.convert("L")
    mean = mean_brightness(image_gray)
    min_max = min_max_contrast(image_gray)
    michelson = michelson_contrast(image_gray)
    rms_contrast = root_mean_square_contrast(image_gray)
    print(f"{mean = }")
    print(f"{min_max = }")
    print(f"{michelson = }")
    print(f"{rms_contrast = }")


if __name__ == "__main__":
    test()
