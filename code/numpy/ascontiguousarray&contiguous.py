"""
array = np.ascontiguousarray(array) 函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
tensor = tensor.contiguous() 函数将一个内存不连续存储的tensor转换为内存连续存储的tensor，使得运行速度更快
"""

import numpy as np
import torch


arr1 = np.arange(0, 12, 1)
print(arr1.flags)
#   C_CONTIGUOUS : True  行连续
#   F_CONTIGUOUS : True  列连续
#   OWNDATA : True
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False


arr2 = arr1.reshape(3, 4)
print(arr2.flags)
#   C_CONTIGUOUS : True  行连续
#   F_CONTIGUOUS : False 列不连续
#   OWNDATA : False
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False


# 转置
arr3 = arr2.T
print(arr3.flags)
#   C_CONTIGUOUS : False 行不连续
#   F_CONTIGUOUS : True  列连续
#   OWNDATA : False
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False


arr4 = arr3[:2, :]
print(arr4.flags)
#   C_CONTIGUOUS : False 行不连续
#   F_CONTIGUOUS : False 列不连续
#   OWNDATA : False
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False

arr5 = np.ascontiguousarray(arr4)
print(arr5.flags)
#   C_CONTIGUOUS : True  行连续
#   F_CONTIGUOUS : False 列不连续
#   OWNDATA : True
#   WRITEABLE : True
#   ALIGNED : True
#   WRITEBACKIFCOPY : False


tensor1 = torch.arange(12)
print(tensor1.is_contiguous())
# True

tensor2 = tensor1.reshape(3, 4)
print(tensor2.is_contiguous())
# True

tensor3 = tensor2.T
print(tensor3.is_contiguous())
# False

tensor4 = tensor3.contiguous()
print(tensor4.is_contiguous())
# True
