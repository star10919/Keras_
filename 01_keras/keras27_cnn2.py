from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D   # 이미지가 2D 이니까


model = Sequential()                                                                    # (N, 10, 10, 1)
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(10, 10, 1)))       # (N, 10, 10, 10) # padding=same 커널사이즈 상관없이 shape이 동일하게 유지됨  # 10 : output   # kernel_size=(2,2) : 가로세로2X2로 자르겠다   # 가로10, 세로10, 흑백이라서 1(컬러면 3(RGB))
model.add(Conv2D(20, (2,2), activation='relu'))                                         # (N, 9, 9, 20)
model.add(Conv2D(30, (2,2), padding='valid'))      # padding 디폴트: valid(패딩X)        # (N, 8, 8, 30)
model.add(MaxPool2D())   # 반으로 줄어듬(연산은 안함)                                  # (N, 4, 4, 30)
model.add(Conv2D(15, (2,2)))                                                            # (N, 3, 3, 15)
model.add(Flatten())  # shape자체가 2차원이 됨(Dense써야 되니까-Dense:2차원)(연산은 안함)   # (N, 135)
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 20)          820
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 30)          2430
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 4, 4, 30)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 15)          1815
_________________________________________________________________
flatten (Flatten)            (None, 135)               0
_________________________________________________________________
dense (Dense)                (None, 64)                8704
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 15,932
Trainable params: 15,932
Non-trainable params: 0
_________________________________________________________________
'''