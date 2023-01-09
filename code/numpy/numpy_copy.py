import numpy as np

a = np.arange(0, 10, 1)
print(a)    # [0 1 2 3 4 5 6 7 8 9]

# 直接复制是引用,会同时修改
b = a
b += 1
print(a)    # [ 1  2  3  4  5  6  7  8  9 10]
print(b)    # [ 1  2  3  4  5  6  7  8  9 10]


a = np.arange(0, 10, 1)
print(a)    # [0 1 2 3 4 5 6 7 8 9]

# copy是深拷贝,完全复制数据
c = a.copy()
c *= 2
print(a)    # [0 1 2 3 4 5 6 7 8 9]
print(c)    # [ 0  2  4  6  8 10 12 14 16 18]
