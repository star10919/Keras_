from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

#train_test_split 으로 만들어라, validation_split 사용
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)


#2. 모델구성(딥러닝 구현)
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))  #통상적으로 loss가 val_loss 보다 좋게 나옴(=> 더 안 좋은 val_loss에 기준을 맞춰서 하파튜해야 함)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.3, shuffle=True)  #train split2번 쓰지말고 validation split으로 써주기

#4. 평가, 예측(훈련데이터와 평가데이터는 같으면 안됨)
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

result = model.predict([11])
print('11의 예측값 :', result)


'''
#5. 결과값
loss:  1.3831898950864008e-12
11의 예측값 : [[11.000001]]
'''

