from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random
import icecream as ic

### train_test_split 사용

#1. 데이터
x = np.array(range(100))   #0~99  100개
y = np.array(range(1, 101))   #1~100   100개

# x_train = x[0:70]
# y_train = y[:70]
# x_test = x[-30:]
# y_test = y[70:]

# print(x_train.shape, y_train.shape)   # (70,) (70,)
# print(x_test.shape, y_test.shape)   # (30,) (30,)


# 무작위 추출로 train, test set 분할  -  train_test_split사용(shuffle=True(디폴트) : 난수 설정)  /  random_state(가급적이면 써줘라) <= 난수표 적용
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)
print(x_test)
print(y_test)


#2. 모델구성(딥러닝 구현)
model = Sequential()
model.add(Dense(9, input_dim=1))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측(훈련데이터와 평가데이터는 같으면 안됨)
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
print('100의 예측값 :', y_predict)


# r2 스코어
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)


'''
# r2 결과값
r2 스코어 : 0.9999996242800373
'''

# y_predict = model.predict(x)

# plt.scatter(x,y)
# plt.plot(x, y_predict, color='red')
# plt.show()
