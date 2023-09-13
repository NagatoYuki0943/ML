from io import StringIO
import numpy as np


# 行列各位一个维度,2行2列为2个维度
s = StringIO("0 1\n2 3")
data = np.loadtxt(s, dtype=np.float32)
print(data)
# [[0. 1.]
#  [2. 3.]]


d = StringIO("M 21 72\nF 35 58")
data = np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
                            'formats': ('S1', 'i4', 'f4')})
print(data)
# [(b'M', 21, 72.) (b'F', 35, 58.)]


c = StringIO("1,0,2\n3,0,4")
x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
print(x)    # [1. 3.]
print(y)    # [2. 4.]


c = StringIO("1,0,2\n3,0,4")
x, y, z = np.loadtxt(c, delimiter=',', usecols=(0, 1, 2), unpack=True)
print(x)    # [1. 3.]
print(y)    # [0. 0.]
print(z)    # [2. 4.]


# The converters argument is used to specify functions to preprocess the text prior to parsing.
# converters can be a dictionary that maps preprocessing functions to each column:
s = StringIO("1.618, 2.296\n3.141, 4.669\n")
conv = {
    0: lambda x: np.floor(float(x)),  # conversion fn for column 0
    1: lambda x: np.ceil(float(x)),  # conversion fn for column 1
}
data = np.loadtxt(s, delimiter=",", converters=conv)
print(data)
# [[1. 3.]
#  [3. 5.]]


# converters can be a callable instead of a dictionary, in which case it is applied to all columns:
s = StringIO("0xDE 0xAD\n0xC0 0xDE")
import functools
conv = functools.partial(int, base=16)
data = np.loadtxt(s, converters=conv)
print(data)
# [[222. 173.]
#  [192. 222.]]

