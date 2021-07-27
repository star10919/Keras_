import numpy as np
from numpy import array
from icecream import ic

# 실습 : 앙상블 모델을 만드시오.
# 결과치 신경쓰지 말고 모델만 완성할 것

# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])

ic(x1.shape, x2.shape, y.shape)   #  x1.shape: (13, 3), x2.shape: (13, 3), y.shape: (13,)
ic(x1_predict.shape, x2_predict.shape)   #  x1_predict.shape: (3,), x2_predict.shape: (3,)

x1 = x1.reshape(13,3,1)
x2 = x2.reshape(13,3,1)
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)

# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, SimpleRNN, LSTM, Input

# 2-1. 모델 1
input1 = Input(shape=(3, 1))
simpleRNN = SimpleRNN(units=32, activation='relu')(input1)
dense = Dense(32, activation='relu')(simpleRNN)
dense = Dense(16, activation='relu')(dense)
dense = Dense(16, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output1 = Dense(1)(dense)

# 2-2. 모델 2
input2 = Input(shape=(3, 1))
simpleRNN = SimpleRNN(units=32, activation='relu')(input2)
dense = Dense(32, activation='relu')(simpleRNN)
dense = Dense(16, activation='relu')(dense)
dense = Dense(16, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output2 = Dense(1)(dense)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(10, name='merge2')(merge1)
merge3 = Dense(5,activation='relu', name='merge3')(merge2)
last_output = Dense(1)(merge3)   #최종 output

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()


# 3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], y, epochs=100, batch_size=1)


# 4. 평가, 예측
# x_input = np.array([5,6,7]).reshape(1,3,1)

y_pred = model.predict([x1_predict, x2_predict])

print('y_pred :', y_pred)


'''
y_pred : [[85.38618]]
'''
