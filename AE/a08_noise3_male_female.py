# keras61_5 남자 여자 데이터에 노이즈를 넣어서 노이즈 제거하시오!!
# 기미, 주근깨, 여드름을 제거하시오

### 노이즈 생성

import numpy as np
from tensorflow.keras.datasets import mnist
from icecream import ic


# 1. 데이터
x_train = np.load('./_save/_npy/k59_5_train_x.npy')
# y_train = np.load('./_save/_npy/k59_5_train_y.npy')
x_test = np.load('./_save/_npy/k59_5_test_x.npy')
# y_test = np.load('./_save/_npy/k59_5_test_y.npy')

ic(x_train.shape, x_test.shape)
# x_train.shape: (2649, 150, 150, 3)
# x_test.shape: (662, 150, 150, 3)



x_train = x_train.reshape(2649, 150, 150, 3).astype('float')/255
x_test = x_test.reshape(662, 150, 150, 3).astype('float')/255

# 일부러 노이즈 생성
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)  # 픽셀에 0에서 0.1의 난수를 더해준다.
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
# 값을 0~1 사이의 값으로 다시 변환
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)      # np.clip : 최솟값을 벗어나면 최솟값으로, 최댓값을 벗어나면 최댓값으로
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)



# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Dropout

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2,2), input_shape=(150, 150, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(1,1))
    model.add(Dropout(0.9))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(Dropout(0.9))
    model.add(MaxPooling2D(1,1))
    model.add(Conv2D(3, (2, 2), activation='sigmoid', padding='same'))
    # model.add(Flatten())
    # model.add(Dropout(0.3))
    # model.add(Dense(units=67500))
    return model


model = autoencoder(hidden_layer_size=32)      # pca 95% : 154

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(3, 5, figsize=(20, 7))


# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3)*255)
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3)*255)
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3)*255)
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()