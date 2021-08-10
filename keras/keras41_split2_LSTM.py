import numpy as np
from icecream import ic

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

x_predict = np.array(range(96, 107))
ic(x_predict.shape)   #  x_predict.shape: (10,)

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
ic(x_pred.shape)    # x_pred.shape: (6, 5)

ic(x.shape, y.shape)   # x.shape: (95, 5), y.shape: (95,)

y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)



# 1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)
ic(x_train.shape, x_test.shape)   #  x_train.shape: (76, 5), x_test.shape: (19, 5)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten

model = Sequential()
model.add(LSTM(units=10, activation='relu', input_shape=(5, 1), return_sequences=True))
model.add(Conv1D(128, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()


# 3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.1, callbacks=[es])
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
    return np.sqrt(mean_squared_error(y_test, y_predict))   # np.sqrt : 루트씌우겠다
rmse = RMSE(y_pred, y_predict)
ic(rmse)



'''
*LSTM + Cov1D
ic| y_predict: array([[100.277534],
                      [101.18534 ],
                      [102.085884],
                      [102.979126],
                      [103.865036],
                      [104.74356 ]], dtype=float32)
걸린시간 : 27.291070222854614
loss : 0.016300853341817856
R2 스코어 :  0.6611314112087712
ic| rmse: 0.994166342876156
'''