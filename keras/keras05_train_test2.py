from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = x[0:7]
y_train = y[:7]
x_test = x[-3:]
y_test = y[7:]

print(x_train)
print(y_train)
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

model.fit(x_train, y_train, epochs=1000, batch_size=1)

#4. 평가, 예측(훈련데이터와 평가데이터는 같으면 안됨)
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

result = model.predict([11])
print('11의 예측값 :', result)


'''
#5. 결과값
loss:  2.2402846298064105e-06
11의 예측값 : [[10.997823]]
'''

# y_predict = model.predict(x)

# plt.scatter(x,y)
# plt.plot(x, y_predict, color='red')
# plt.show()
