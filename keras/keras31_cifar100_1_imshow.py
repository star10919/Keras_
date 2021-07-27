from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)

ic(x_train[27])
print('y[27] 값 :', y_train[27])   # [52]

plt.imshow(x_train[27])
plt.show()