from tensorflow.keras.models import Sequential  # Sequential : 순차적으로 내려가는 모델
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

### 딥러닝

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,6,7,8,9,10]) 

#2. 모델구성(딥러닝 구현)
model = Sequential()
model.add(Dense(5, input_dim=1))     # 앞이 output, 뒤가 input
model.add(Dense(4))   # hidden layer #(node개수) # input 적지 않음(위의 output이 아래의 input이 되니까)
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))   # output layer

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import TensorBoard
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x, y, epochs=50, batch_size=1, callbacks=[tb], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict([6])
print('6의 예측값 :', result)


'''
#5. 결과값
loss:  0.38008081912994385
6의 예측값 : [[5.7160087]]
'''

y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict, color='red')
plt.show()