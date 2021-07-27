from tensorflow.keras.datasets import cifar10   #  32, 32, 3(color)
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

### 컬러(a, b, c, 3)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)

ic(x_train[5])
print('y[5]의 값 :', y_train[5])   # [1]

plt.imshow(x_train[5])
plt.show()
