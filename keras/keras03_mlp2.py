'''
1. [1,2,3]   스칼라 3개 = (3, ) [[1],[2]]
2. [[1,2,3]]   1행 3열 / =3개의 특성을 가진 1개의 데이터
3. [[1],[2],[3]]   3행 1열 /
4. [[1,2],[3,4],[5,6]]   3행 2열 /
5. [[[1,2,3],[4,5,6]]]   1면 2행 3열
6. [[[1,2],[3,4],[5,6]]]   1면 3행 2열
7. [[[1],[2]],[[3],[4]]]   2면 2행 1열

제일 바깥에 있는 괄호는 무시하고 계산하기
가장 작은단위 안에 원소의 개수 : 열
작은단위의 개수 : 행
특성 = 피쳐 = 컬럼 = 열
행무시, 열우선
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

#1-1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]])   # (3,10)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])   # (10,) = 10 스칼라
print(x.shape)   
print(y.shape)   

#1-2. 행열바꾸기
x = np.transpose(x)
print(x.shape)   #(10, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
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
loss : 2.1821415785439058e-08
10, 1.3, 1의 예측값 : [[20.00019]]
'''

y_predict = model.predict(x)

plt.scatter(x,[y,y,y])
plt.plot(y_predict, color='red')
plt.show()