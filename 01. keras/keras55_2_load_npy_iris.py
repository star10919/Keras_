import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

### 데이터 로드하기

x_data_iris = np.load('./_save/_npy/k55_x_data_iris.npy')
y_data_iris = np.load('./_save/_npy/k55_y_data_iris.npy')

ic(x_data_iris)
ic(y_data_iris)
ic(x_data_iris.shape, y_data_iris.shape)        # x_data_iris.shape: (150, 4), y_data_iris.shape: (150,)

# np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris)
# np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data_iris, y_data_iris, train_size=0.7, random_state=9)

# 1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ic(x_train.shape, x_test.shape)   #  x_train.shape: (105, 4), x_test.shape: (45, 4)
x_train = x_train.reshape(105, 4, 1)
x_test = x_test.reshape(45, 4, 1)

# 1-3. y 데이터 전처리
from tensorflow.keras.utils import to_categorical    # 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
ic(y_data_iris[:5])
# [0,0,0,0,0]
# [[1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]]
ic(y_data_iris.shape)   # (150, 3)


# 2. 모델
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(4,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # softmax : 다중분류   # 0,1,2  3개라서 3개로 나와야 함(150, 3)


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # binary_crossentropy : 2진 분류   # metrics(결과에 반영X):평가지표

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])
end = time.time() - start


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)   # 무조건 낮을수록 좋다
print('걸린시간 :', end)
print('loss :', results[0])
print('accuracy :', results[1])

ic(y_test[:5])
y_predict = model.predict(x_test[:5])
ic(y_predict)   # 소프트맥스 통과한 값


'''
loss : 0.03401293233036995
accuracy : 1.0

*cnn + Flatten
걸린시간 : 3.87078595161438
loss : 0.022757794708013535
accuracy : 1.0

*cnn + GAP
걸린시간 : 3.823838233947754
loss : 0.13620592653751373
accuracy : 0.9555555582046509

*LSTM
걸린시간 : 6.369047403335571
loss : 0.06772168725728989
accuracy : 1.0

*load_npy
걸린시간 : 5.072533845901489
loss : 0.07786381244659424
accuracy : 0.9777777791023254
'''