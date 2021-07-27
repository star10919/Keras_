import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from icecream import ic

# 결과값이 80 근접하게 튜닝하시오.

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
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))   # units - output node 개수  # input_shape 에는 항상 데이터의 개수(맨 앞) 무시하고 쓰기
model.add(GRU(units=32, activation='relu', input_shape=(3, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dense(13, activation='relu'))
# model.add(Dense(26, activation='relu'))
model.add(Dense(10, activation='relu'))
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
[[80.300835]]
'''
