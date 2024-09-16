import numpy as np
import torch


a = np.array(
    [
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
    ]
)


# 默认情况下展平数据,将所有数据当做单个值进行unique
print(np.unique(a))
# [0 1]
print(torch.unique(torch.from_numpy(a)).numpy())
# [0 1]


# 指定维度含义是以指定维度为独立数据,可以是vector,maxrix
# axis = dim = 0 含义是将 axis=0 方向的每个值当做独立数据
print(np.unique(a, axis=0))
# [[[0 0 1 1]
#   [0 0 1 1]
#   [1 1 1 1]]
#  [[1 1 0 0]
#   [1 1 0 0]
#   [0 0 1 1]]]
print(torch.unique(torch.from_numpy(a), dim=0).numpy())
# [[[0 0 1 1]
#   [0 0 1 1]
#   [1 1 1 1]]
#  [[1 1 0 0]
#   [1 1 0 0]
#   [0 0 1 1]]]


# 指定维度含义是以指定维度为独立数据,可以是vector,maxrix
# axis = dim = 1 含义是将 axis=1 方向的每个值当做独立数据
print(np.unique(a, axis=1))
# [[[0 0 1 1]
#   [1 1 0 0]]
#  [[1 1 1 1]
#   [0 0 1 1]]
#  [[0 0 1 1]
#   [1 1 0 0]]]
print(torch.unique(torch.from_numpy(a), dim=1).numpy())
# [[[0 0 1 1]
#   [1 1 0 0]]
#  [[1 1 1 1]
#   [0 0 1 1]]
#  [[0 0 1 1]
#   [1 1 0 0]]]


# 指定维度含义是以指定维度为独立数据,可以是vector,maxrix
# axis = dim = 2 含义是将 axis=2 方向的每个值当做独立数据
print(np.unique(a, axis=2))
# [[[0 1]
#   [0 1]
#   [1 0]]
#  [[1 0]
#   [1 0]
#   [1 1]]
#  [[0 1]
#   [0 1]
#   [1 0]]]
print(torch.unique(torch.from_numpy(a), dim=2).numpy())
# [[[0 1]
#   [0 1]
#   [1 0]]
#  [[1 0]
#   [1 0]
#   [1 1]]
#  [[0 1]
#   [0 1]
#   [1 0]]]
