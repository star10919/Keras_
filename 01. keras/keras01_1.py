from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([1,2,3])  # 스칼라3 벡터1-1차원임
y = np.array([1,2,3]) 

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))  #Dense(output(y) - Output node 2개, input(x)_dim=1 - Input node 1개)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # 모델이 알아먹도록 compile! # mse방식으로 로스를 줄이겠다!  #optimizer='adam  평타85%로 나옴

model.fit(x, y, epochs=500, batch_size=1)  # 훈련시키겠다!  # epochs=500 : 훈련500번시키겠다  # batch_size=1 : 원소 1개씩 넣어서 훈련시키겠다.

#4. 평가, 예측
loss = model.evaluate(x, y) # loss='mse' 값이 반환됨 # 원래는 데이터와 다른 값을 넣어서 평가함
print('loss: ', loss)

result = model.predict([4])  # 스칼라1 벡터1-1차원임  # predict은 fit에서 생성된 w, b가 들어가 있음
print('4의 예측값 :', result)  # 하이퍼파라미터튜닝


'''
#5. 결과값
loss:  5.696184084309774e-13
4의 예측값 : [[4.000002]]
'''

y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict, color='red')
plt.show()