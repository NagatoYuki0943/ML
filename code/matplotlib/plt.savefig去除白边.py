
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


array = np.random.uniform(0, 255, (224, 224, 3)).astype(np.uint8)
height, width = array.shape[:2]
image = Image.fromarray(array)

dpi = 100
# 这样还是不能保存原图大小
plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
plt.imshow(image)
plt.axis("off")
plt.savefig("uniform.jpg", bbox_inches='tight', pad_inches = 0) # 后面2个参数去除边框
plt.show()
plt.close()
