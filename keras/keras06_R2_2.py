import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from icecream import ic

# 과제 2
# R2를 0.9까지로 올려라!!!

#1.데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

#2.모델
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=1))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=2)

#4.평가, 예측
loss = model.evaluate(x, y)
print('loss:', loss)

# y_predict = model.predict([6])
# print('6의 예측값 :', y_predict)

'''
#5. 결과값
loss: 0.270074725151062
R2 스코어 : 0.8649626361929663
'''

# R2 결정 계수 : 정확도와 유사한 지표
y_predict = model.predict(x)
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
print('R2 스코어 :', r2)

