import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from icecream import ic

### LSTM(3차원 -> 2차원) 2번 쓰려면?  return_sequences=True(차원유지 -> 차원유지) 사용!         # 통상적으로 LSTM 2번씩 쓰는 경우는 많지 않음

# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])   # => (None, 3, 1) 로 reshape 해 줘야 함



ic(x.shape, y.shape)   #  x.shape: (13, 3), y.shape: (13,)

x = x.reshape(13, 3, 1)   # (batch_size, timesteps, feature)   *feature : 몇 개씩 자르는지(가장 작은 연산의 단위)
x = x.reshape(x.shape[0], x.shape[1],1)
x_predict = x_predict.reshape(1, x_predict.shape[0], 1)


# 2. 모델구성
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(3, 1), return_sequences=True))   # return_sequences - default:False
model.add(LSTM(units=16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()


# 3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


# 4. 평가, 예측
# x_input = np.array([5,6,7]).reshape(1,3,1)

results = model.predict(x_predict)
print(results)

'''
[[80.18278]]
'''

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480             #원래는 (N, 10)-2차원이 되었어야 함 / return_sequences=True 사용해서 받아진 차원그대로  나와짐.
_________________________________________________________________                                          3차원으로 나와짐.(LSTM 또 사용가능해짐.)
lstm_1 (LSTM)                (None, 7)                 504             #LSTM 하나 더 받음
_________________________________________________________________
dense (Dense)                (None, 5)                 40
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6
=================================================================
Total params: 1,030
Trainable params: 1,030
Non-trainable params: 0
_________________________________________________________________
'''
