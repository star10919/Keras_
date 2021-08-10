import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from icecream import ic

### RNN - 3차원으로 들어가서 2차원으로 나옴(그래서 바로 Dense로 받아줄 수 가 있음)

# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

ic(x.shape, y.shape)   #  x.shape: (4, 3), y.shape: (4,)

x = x.reshape(4, 3, 1)   # (batch_size, timesteps, feature)   *feature : 몇 개씩 자르는지(가장 작은 연산의 단위)


# 2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))   # units - output node 개수  # input_shape 에는 항상 데이터의 개수(맨 앞) 무시하고 쓰기
model.add(SimpleRNN(10, activation='relu', input_length=3, input_dim=1))   # 동일한 표현
                                            # timesteps     # feature
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
simple_rnn (SimpleRNN)       (None, 10)                120
_________________________________________________________________
dense (Dense)                (None, 100)               1100
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1010
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
_________________________________________________________________
dense_3 (Dense)              (None, 10)                110
_________________________________________________________________
dense_4 (Dense)              (None, 10)                110
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 2,571
Trainable params: 2,571
Non-trainable params: 0
_________________________________________________________________

 => parameter 가 120 인 이유 : (Input + bias) * 10 + output * output = (Input + bias + output) * output
'''


# 3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


# 4. 평가, 예측
x_input = np.array([5,6,7]).reshape(1,3,1)

results = model.predict(x_input)
print(results)


'''
[[7.9656234]]
[[8.033126]]
[[8.004619]]
[[7.9767957]]
'''
