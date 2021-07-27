from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten   # 이미지가 2D 이니까

### Flatten - Dense 에 넣기 전에 2차원으로 바꿔줘야 함

model = Sequential()                                                        # (N, 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5, 5, 1)))             # (N, 4, 4, 10)    # 10 : output    # kernel_size=(2,2) : 가로세로2X2로 자르겠다   # 가로10, 세로10, 흑백이라서 1(컬러면 3(RGB))
model.add(Conv2D(20, (2,2), activation='relu'))                             # (N, 3, 3, 20)
model.add(Conv2D(30, (2,2)))
model.add(Flatten())  # shape자체가 2차원이 됨(Dense써야 되니까-Dense:2차원)   # (N, 180)
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()