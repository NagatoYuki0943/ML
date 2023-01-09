import numpy as np
import torch


def sigmoid(x: np.ndarray) -> np.ndarray:
    """sigmoid
        Sigmoid(x) = \frac {1} {1 + e^{-x}}
    Args:
        x (np.ndarray): data

    Returns:
        np.ndarray: sigmoid result
    """
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray, axis: int=0) -> np.ndarray:
    """将每个值求e的指数全都变为大于0的值，然后除以求指数之后的总和
        Softmax(x)_i = \frac {e^{x_i}} {{\sum_{j=1}^{n}} e^{x_j}}
    Args:
        x (np.ndarray): data
        axis (int, optional): 在哪个通道上计算. Defaults to 0.

    Returns:
        np.ndarray: softmax result
    """
    # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)


if __name__ == "__main__":
    x = np.array([1., 2., 3., 4., 5.])
    print(x)                # [1. 2. 3. 4. 5.]

    print('\nsigmoid:')
    print(sigmoid(x))       # [0.73105858 0.88079708 0.95257413 0.98201379 0.99330715]
    y = torch.sigmoid(torch.as_tensor(x))
    print(y.numpy())        # [0.73105858 0.88079708 0.95257413 0.98201379 0.99330715]

    print('\nSoftmax:')
    print(softmax(x))       # [0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]
    y = torch.softmax(torch.as_tensor(x), dim=0)
    print(y.numpy())        # [0.01165623 0.03168492 0.08612854 0.23412166 0.63640865]

    x = np.array([
        [2.3, 1.5, -3.1, 5.0, 1.1],
        [0.8, 5.4, 6.1, -3.3, 4.0],
    ])
    print(softmax(x, axis=1))
    # [[6.01150428e-02 2.70114299e-02 2.71514457e-04 8.94495710e-01
    #   1.81063029e-02]
    #  [3.07342194e-03 3.05757279e-01 6.15719548e-01 5.09348242e-05
    #   7.53988166e-02]]
    y = torch.softmax(torch.as_tensor(x), dim=1)
    print(y.numpy())
    # [[6.01150428e-02 2.70114299e-02 2.71514457e-04 8.94495710e-01
    #   1.81063029e-02]
    #  [3.07342194e-03 3.05757279e-01 6.15719548e-01 5.09348242e-05
    #   7.53988166e-02]]
