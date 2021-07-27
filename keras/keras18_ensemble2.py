import numpy as np
from icecream import ic

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1,101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.array(range(1001, 1101))
y2 = np.array(range(1901, 2001))

# ic(x1.shape, x2.shape, y.shape)    # x1.shape: (100, 3),    x2.shape: (100, 3),    y1.shape: (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2, train_size=0.7)

ic(x1_train.shape, x1_test.shape,       # (70, 3) , (30, 3)
    x2_train.shape, x2_test.shape,      # (70, 3) , (30, 3)
    y1_train.shape, y1_test.shape,      # (70,) , (30,)
    y2_train.shape, y2_test.shape,)     # (70,) , (30,)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(3, activation='relu', name='dense2')(dense1)
dense3 = Dense(2, activation='relu', name='dense3')(dense2)
output1 = Dense(3, name='output1')(dense3)  #히든레이어

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(4, activation='relu', name='dense11')(input2)
dense12 = Dense(4, activation='relu', name='dense12')(dense11)
dense13 = Dense(4, activation='relu', name='dense13')(dense12)
dense14 = Dense(4, activation='relu', name='dense14')(dense13)
output2 = Dense(4, name='output2')(dense14)  #히든레이어

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1,output2])   #연결만 한거임
merge2 = Dense(10, name='merge2')(merge1)
merge3 = Dense(5, activation='relu', name='merge3')(merge2)    #분기
# last_output = Dense(1)(merge3)   #최종 output

output21 = Dense(7)(merge3)
last_output1 = Dense(1, name='last_output1')(output21)

output22 = Dense(8)(merge3)
last_output2 = Dense(1, name='last_output2')(output22)


model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train,y2_train], epochs=100, batch_size=8, verbose=1)

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], [y1_test,y2_test])
ic(results)

'''
results: [117.58346557617188,       # last_output1_loss + last_output2_loss
              62.05646896362305,    # last_output1_loss
              55.52699279785156,    # last_output2_loss
              6.632175922393799,    # last_output1_mae
              6.286913871765137]    # last_output2_mae
'''
