import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from icecream import ic

# 과제 2
# 함수형으로 리폼하시오
# 서머리로 확인
# R2를 0.9까지로 올려라!!!

#1.데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

#2.모델
input1 = Input(shape=(1,))
dense1 = Dense(60)(input1)
dense2 = Dense(50)(dense1)
dense3 = Dense(30)(dense2)
dense4 = Dense(24)(dense3)
dense5 = Dense(36)(dense4)
dense5 = Dense(15)(dense4)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

model.summary()


# model = Sequential()
# model.add(Dense(6, input_dim=1))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(24))
# model.add(Dense(36))
# model.add(Dense(15))
# model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=20000)

#4.평가, 예측
loss = model.evaluate(x, y)
print('loss:', loss)

# y_predict = model.predict([6])
# print('6의 예측값 :', y_predict)

'''
#5. 결과값

'''

# R2 결정 계수 : 정확도와 유사한 지표
y_predict = model.predict(x)
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
print('R2 스코어 :', r2)

