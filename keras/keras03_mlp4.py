from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

#1-1. 데이터
x = np.array([range(10)])   # (1,10)
x = np.transpose(x)  # (10,1)

y = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]])  # (3,10)
y = np.transpose(y)   # (10,3)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
 
result = model.predict([[9]])
print('0, 21, 201의 예측값 :', result)


'''
#5. 결과값
loss : 0.005478133447468281
0, 21, 201의 예측값 : [[9.98473    1.5425142  0.97333896]]
'''

y_predict = model.predict(x)

plt.scatter([x,x,x],y)
plt.plot(y_predict, color='red')
plt.show()