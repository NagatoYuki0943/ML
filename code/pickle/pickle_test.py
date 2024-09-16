import pickle
import numpy as np


x = np.arange(6).reshape(2, 3)


# 保存文件, mode为b
with open("pickle.pkl", mode="wb") as f:
    pickle.dump(x, f)


# pickle读文件, mode为b
with open("pickle.pkl", mode="rb") as f:
    y = pickle.load(f)


print(np.all(x == y))  # True
