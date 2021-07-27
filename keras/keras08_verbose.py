from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import time

#1-1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# print(x.shape)      # (3, 10)
# print(y.shape)      # (10,)

#1-2. 행열바꾸기
x = np.transpose(x)
# print(x.shape)   #(10, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x, y, epochs=200, batch_size=1, verbose=3)
end = time.time() - start
ic("걸린시간 :", end)
#verbose=0 결과만 보임  /   2.350717544555664
#verbose=1 프로그레스 바, Epoch, 로스값  /   3.5974180698394775    verbose=1 일때, batch=1, 10 시간차이
#verbose=2 Epoch, 로스값  /   2.6877856254577637
#verbose=3 Epoch  /   2.547189712524414


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([[10, 1.3, 1]])
print('10, 1.3, 1의 예측값 :', result)


'''
#5. 결과값
loss : 0.010416761972010136
10, 1.3, 1의 예측값 : [[19.804848]]
'''

y_predict = model.predict(x)
