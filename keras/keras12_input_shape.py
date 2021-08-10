from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1-1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], [10,9,8,7,6,5,4,3,2,1]])   # (3,10)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])   # (10,)  => 스칼라10개(=벡터1개)
print(x.shape)   
print(y.shape)   

#1-2. 행열바꾸기
x = np.transpose(x)
print(x.shape)   #(10, 3)  행무시 열우선

#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=3))
model.add(Dense(10, input_shape=(3,)))   #행무시 열우선 /   2차원 : (10, 3)=>  input_dim=3  =  input_shape=(3,)
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([[10, 1.3, 1]])
print('10, 1.3, 1의 예측값 :', result)


'''
#5. 결과값
loss : 3.21105403600086e-06
10, 1.3, 1의 예측값 : [[20.00213]]
'''

y_predict = model.predict(x)
