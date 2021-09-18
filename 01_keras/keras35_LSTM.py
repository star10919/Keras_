import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from icecream import ic

### LSTM - 시계열 데이터 예측(RNN 과 다른 점 : 연산방식이 다름) / 히든스테이트 4개

# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

ic(x.shape, y.shape)   #  x.shape: (4, 3), y.shape: (4,)

x = x.reshape(4, 3, 1)   # (batch_size, timesteps, feature)   *feature : 몇 개씩 자르는지(가장 작은 연산의 단위)


# 2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))   # units - output node 개수  # input_shape 에는 항상 데이터의 개수(맨 앞) 무시하고 쓰기
model.add(LSTM(units=10, activation='relu', input_shape=(3, 1)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 10)                110
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 1,141
Trainable params: 1,141
Non-trainable params: 0
_________________________________________________________________

 => parameter 가 480 인 이유 : 
'''


# 3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


# 4. 평가, 예측
x_input = np.array([5,6,7]).reshape(1,3,1)

results = model.predict(x_input)
print(results)


'''
[[8.544802]]
'''
