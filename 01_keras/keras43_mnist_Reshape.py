import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from icecream import ic

### 모델링 중간중간에 shape 변경 가능(Reshape 사용)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

ic(x_train.shape, y_train.shape)   # x_train.shape: (60000, 28, 28), y_train.shape: (60000,)
ic(x_test.shape, y_test.shape)     # x_test.shape: (10000, 28, 28), y_test.shape: (10000,)


'''
x_train = x_train.reshape(60000, 28, 28, 1)   # reshape : 내용물과 순서는 바뀌지 않음(차원만 바뀜)  -  총 곱셈이 같아야 함.
x_test = x_test.reshape(10000, 28, 28, 1)   # 컨벌루션은 무조건 4차원으로 만들어서 넣어줘야 함.
'''
ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


# #1-2. y 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)
# ic(y_train.shape, y_test.shape)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape

model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(4,4), padding='same', input_shape=(28, 28, 1)))
model.add(Dense(units=10, activation='relu', input_shape=(28, 28)))   # Dense로도 3, 4차원 데이터 받아줄 수 있음
model.add(Flatten())        # (N, 280)
model.add(Dense(784))       # (N, 784)
model.add(Reshape((28, 28, 1)))         #(N, 28, 28, 1)
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(64, (2,2)))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(10, activation='softmax'))

model.summary()


# 3. 컴파일(ES), 훈련           metrics['acc]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.0012, callbacks=[es])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('category :', results[0])
print('accruacy :', results[1])

y_predict = model.predict(x_test)
# R2
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
ic(r2)

# RMSE
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # np.sqrt : 루트씌우겠다
rmse = RMSE(y_test, y_predict)
ic(rmse)


# acc로만 판단해보자.
# 0.98 이상 완성


'''
*cnn
category : 0.04307371377944946
accruacy : 0.9900000095367432

*dnn
category : 0.15464764833450317
accruacy : 0.954800009727478
ic| r2: 0.9226106878077964
ic| rmse: 0.08323233681168003

*dnn + reshape
category : 0.19469135999679565
accruacy : 0.9693999886512756
ic| r2: 0.9423217721434817
ic| rmse: 0.071905070480354
'''
