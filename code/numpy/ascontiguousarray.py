"""
ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
"""

import numpy as np


a = np.arange(0, 12, 1)
print(a.flags)
#   C_CONTIGUOUS : True  行连续
#   F_CONTIGUOUS : True  列连续
#   OWNDATA : True
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False


b = a.reshape(3, 4)
print(b.flags)
#   C_CONTIGUOUS : True  行连续
#   F_CONTIGUOUS : False 列不连续
#   OWNDATA : False
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False


# 转置
c = b.T
print(c.flags)
#   C_CONTIGUOUS : False 行不连续
#   F_CONTIGUOUS : True  列连续
#   OWNDATA : False
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False


d = c[:2, :]
print(d.flags)
#   C_CONTIGUOUS : False 行不连续
#   F_CONTIGUOUS : False 列不连续
#   OWNDATA : False
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False

e = np.ascontiguousarray(d)
print(e.flags)
#   C_CONTIGUOUS : True  行连续
#   F_CONTIGUOUS : False 列不连续
#   OWNDATA : True
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False
