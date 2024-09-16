import numpy as np


# 单维数组
x = np.arange(5)
print(x)  # [0 1 2 3 4]

# 单个下标
y = np.delete(x, 2, axis=0)
print(x)  # [0 1 2 3 4] 不会修改原数组
print(y)  # [0 1 3 4]

# 列表下标
y = np.delete(x, [0, 2], axis=0)
print(y)  # [1 3 4]


# 多维数组
x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
print(x)
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

y = np.delete(x, 1, axis=0)
print(y)
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]]

y = np.delete(x, [1, 2], axis=1)
print(y)
# [[[ 0  1  2  3]]
#  [[12 13 14 15]]]

y = np.delete(x, [1, 2], axis=2)
print(y)
# [[[ 0  3]
#   [ 4  7]
#   [ 8 11]]
#  [[12 15]
#   [16 19]
#   [20 23]]]
