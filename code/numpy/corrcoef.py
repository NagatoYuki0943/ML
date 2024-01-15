"""
计算相关性系数
在NumPy中也提供了相关系数计算函数corrcoef可用于快速计算两个数组之间的相关系数，
numpy.corrcoef()函数返回的是一个2x2的相关性矩阵，其中对角线元素是自身的相关性（总是1），非对角线元素是两个变量之间的相关性。

注意：这里的相关性是皮尔逊相关性，测量的是两个变量之间的线性关系。相关性值的范围是-1到1，-1表示完全的负相关，1表示完全的正相关，0表示无相关性。
"""

import numpy as np


weight = np.array([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
fuel_efficiency = np.array([30, 25, 22, 20, 18, 16, 15, 13])

# 计算相关性矩阵
correlation_matrix = np.corrcoef(weight, fuel_efficiency)

print(correlation_matrix)
# [[ 1.        -0.9767344]
#  [-0.9767344  1.       ]]

# 这是一个2x2的相关性矩阵。在这个矩阵中：
# 第一行第一列的数值1代表的是weight（重量）与自身的相关性，这个值永远都是1，因为任何变量与自身的相关性总是最大的。
# 第二行第二列的数值1同样代表的是fuel_efficiency（燃油效率）与自身的相关性，同样这个值也是1。
# 第一行第二列的数值-0.9767344代表的是weight与fuel_efficiency之间的相关性。这个数值非常接近-1，表示weight与fuel_efficiency之间存在非常强烈的负相关性。也就是说，weight增加，fuel_efficiency就会减少。
# 第二行第一列的数值-0.9767344同样代表的是fuel_efficiency与weight之间的相关性，它与第一行第二列的数值完全相同，因为相关性是无方向的，A与B的相关性等同于B与A的相关性。
