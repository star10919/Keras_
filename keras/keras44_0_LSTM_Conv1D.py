from re import S
import numpy as np
from icecream import ic
from sklearn.utils import shuffle
from tensorflow.python.keras.layers.core import Flatten

'''
#실습 1~100까지의 데이터를
        x                 y
1, 2, 3, 4, 5             6
...
95, 96, 97, 98, 99       100
'''

# 1. 데이터
a = np.array(range(1, 101))
size = 6

x_predict = np.array(range(96, 105))
ic(x_predict.shape)   #  x_predict.shape: (9,)

'''
        x                 y
96, 97, 98, 99, 100       ?
...
101, 102, 103, 104, 105   ?

예상 결과값 : 101, 102, 103, 104, 105, 106
평가지표 : RMSE, R2
'''


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)
dataset2 = split_x(x_predict, size)

print("dataset :\n", dataset)

x = dataset[:, :-1]
y = dataset[:, -1]
x_pred = dataset2[:, :-1]
y_pred = dataset2[:, -1]
ic(x_pred)
ic(x_pred)


print("x :\n", x.shape)   # (95, 5)
print("y :", y.shape)     # (95,)
print("x_pred :\n", x_pred.shape)  # (4, 5)
print("y_pred :\n", y_pred.shape)  # (4,)


y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=9)
x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, train_size=0.9, shuffle=True, random_state=9)


# 1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)
ic(x_train.shape, x_val.shape)   # x_train.shape: (9, 5), x_val.shape: (1, 5), x_test.shape: (10, 5)


x_train = x_train.reshape(9, 5, 1)
x_val = x_val.reshape(1, 5, 1)
x_test = x_test.reshape(10, 5, 1)
x_pred = x_pred.reshape(4, 5, 1)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D

model = Sequential()
# model.add(LSTM(units=10, activation='relu', input_shape=(5, 1)))
model.add(Conv1D(10, 2, input_shape=(5, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=5, callbacks=[es])
end = time.time() - start


# 4. 평가, 예측
result = model.evaluate(x_test, y_test)

y_predict = model.predict(x_pred)
ic(y_predict)

print('걸린시간 :', end)
print('loss :', result)
# R2
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_pred, y_predict)
print('R2 스코어 : ', r2)

# RMSE
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_pred, y_predict))   # np.sqrt : 루트씌우겠다
rmse = RMSE(y_test, y_predict)
ic(rmse)



'''
loss : 0.5054280757904053
R2 스코어 :  0.8096759070991538
ic| rmse: 0.4877551805220091

*Conv1D
걸린시간 : 12.419901132583618
loss : 1.1263048648834229
R2 스코어 :  -233.0424494003295
ic| rmse: 17.10418258059741
'''