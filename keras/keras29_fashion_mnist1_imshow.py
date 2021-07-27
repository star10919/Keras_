from scipy.sparse import data
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

### 흑백

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

ic(x_train[1220])
ic(y_train[1220])

plt.imshow(x_train[1220], 'gray')
plt.show()