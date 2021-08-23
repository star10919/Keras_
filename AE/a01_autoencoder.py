### 앞뒤가 똑같은 오~토인코더~(중요하지 않은 특성들은 도태됨)    / (특징이 강한 것을 더 강하게 해주는 것은 아님)

import numpy as np
from tensorflow.keras.datasets import mnist


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255


# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))                     # 784로 들어가서
# encoded = Dense(64, activation='relu')(input_img)   # 중요하지 않은 컬럼 삭제되고
encoded = Dense(1064, activation='relu')(input_img)   # 확 늘림

decoded = Dense(784, activation='sigmoid')(encoded) # 다시 784로 나옴     # 최소 0, 최대 1        *** 결과값이 제일 좋음
# decoded = Dense(784, activation='relu')(encoded)                         # 최소 0, 최대 무한
# decoded = Dense(784, activation='linear')(encoded)                       # 최소 무한, 최대 무한
# decoded = Dense(784, activation='tanh')(encoded)                         # 최소 -1, 최대 1


autoencoder = Model(input_img, decoded)

# autoencoder.summary()


# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)      # y 필요없음 / x 넣어서  x 나옴(앞뒤가 똑같은 오토인코더)


# 4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)




# 시각화
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()





