# 2번 카피해서 복붙
# (CNN으로))딥하게 구성
# 2개의 모델을 구성하는데 하나는 기본적 오토인코더, 다른 하나는 딥하게 구성
'''
Conv2d
MaxPool
Conv2D
MaxPool
Conv2D -> encoder

Conv2D
UpSampling2D
Conv2D
UpSampling2D
Conv2D
UpSampling2D
Conv2D(1, )     -> Decoder
'''

### 앞뒤가 똑같은 오~토인코더~(중요하지 않은 특성들은 도태됨)    / (특징이 강한 것을 더 강하게 해주는 것은 아님)

import numpy as np
from tensorflow.keras.datasets import mnist


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float')/255


# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D

def autoencoder(hidden_layer_size1, hidden_layer_size2, kernel_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size1, kernel_size, input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPool2D(hidden_layer_size2))
    model.add(Conv2D(hidden_layer_size1, kernel_size, activation='relu'))
    model.add(MaxPool2D(hidden_layer_size2))
    model.add(Conv2D(hidden_layer_size1, kernel_size, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model = autoencoder(512, (2,2), (2,2))      # pca 95% : 154
model.add(Conv2D(8, (2,2), padding='same'))
model.add(UpSampling2D(2,2))
model.add(Conv2D(4, (2,2), padding='same'))
model.add(UpSampling2D(4,4))
model.add(Conv2D(2, (7,7), padding='same'))
model.add(UpSampling2D(2,2))
model.add(Conv2D((1,)))



model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))


# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()