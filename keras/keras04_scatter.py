from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])   # 10스칼라(=1벡터)
y = np.array([1,2,4,3,5,7,9,3,8,12])   # 10스칼라(=1벡터)

#2. 모델구성(딥러닝 구현)
model = Sequential()
model.add(Dense(9, input_dim=1))     # 앞이 output, 뒤가 input
model.add(Dense(8))   # hidden layer #(node개수) # input 적지 않음(위의 output이 아래의 input이 되니까)
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))   # output layer

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=3000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict([11])
print('11의 예측값 :', result)


'''
#5. 결과값
loss:  3.6655948162078857
11의 예측값 : [[10.612054]]
'''

y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x, y_predict, color='red')
plt.show()
