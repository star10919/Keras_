from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

### validation data

#1. 데이터(훈련데이터와 평가데이터를 나누는 이유 : 과적합에 걸리지 않기 위해서)
x_train = np.array([1,2,3,4,5,6,7])   # 훈련, 공부
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])  # 평가 데이터(30%)
y_test = np.array([8,9,10])
x_val = np.array([11,12,13])   # 문제집풀기
y_val = np.array([11,12,13])

#2. 모델구성(딥러닝 구현)
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))  #통상적으로 loss가 val_loss 보다 좋게 나옴(=> 더 안 좋은 val_loss에 기준을 맞춰서 하파튜해야 함)

#4. 평가, 예측(훈련데이터와 평가데이터는 같으면 안됨)
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

result = model.predict([11])
print('11의 예측값 :', result)


'''
#5. 결과값
loss : 9.852859630607447e-13
11의 예측값 : [[10.999999]]
'''

# y_predict = model.predict(x)
