import numpy as np
from icecream import ic

# 소스 완성(input모델 1 -> output모델 2)

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1,101)])
# x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
# x2 = np.transpose(x2)
y1 = np.array(range(1001, 1101))
y2 = np.array(range(1901, 2001))

# ic(x1.shape, x2.shape, y.shape)    # x1.shape: (100, 3),    x2.shape: (100, 3),    y1.shape: (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, train_size=0.7)

ic(x1_train.shape, x1_test.shape,       # (70, 3) , (30, 3)
    y1_train.shape, y1_test.shape,      # (70,) , (30,)
    y2_train.shape, y2_test.shape,)     # (70,) , (30,)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 앙상블 모델1
input1 = Input(shape=(3,))
dense1 = Dense(50, activation='relu', name='dense1')(input1)
dense2 = Dense(30, activation='relu', name='dense2')(dense1)
dense3 = Dense(20, activation='relu', name='dense3')(dense2)
output1 = Dense(30, name='output1')(dense3)  #히든레이어

#2-2. 모델 분기1
output21 = Dense(10)(output1)
dense21 = Dense(5, activation='relu', name='dense21')(output21)
last_output1 = Dense(1, name='last_output1')(dense21)

#2-3. 모델 분기2
output22 = Dense(10)(output1)
dense31 = Dense(5, activation='relu', name='dense31')(output22)
last_output2 = Dense(1, name='last_output2')(dense31)


model = Model(inputs=input1, outputs=[last_output1, last_output2])

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8, verbose=1)

#4. 평가, 예측
results = model.evaluate(x1_test, [y1_test, y2_test])
ic(results)
