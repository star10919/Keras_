from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

#1-1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])   # (3,10)    # range(10):0~9 / range(21,31):21~30  / range(201,211):201~210
x = np.transpose(x)  # (10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]])  # (3,10)
y = np.transpose(y)  # (10,3)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
 
result = model.predict([[10, 21, 201]])
print('0, 21, 201의 예측값 :', result)


'''
#5. 결과값
loss : 0.006024095229804516
0, 21, 201의 예측값 : [[10.498683    1.6212702   0.34287584]]
'''

y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(y_predict, color='red')
plt.show()