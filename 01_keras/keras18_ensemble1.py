import numpy as np
from icecream import ic

### concatenate(메소드), Concatenate(클래스) - 앙상블((함수형)모델 합치기)

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1,101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.array(range(1001, 1101))

# ic(x1.shape, x2.shape, y.shape)    # x1.shape: (100, 3),    x2.shape: (100, 3),    y1.shape: (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True, random_state=66)

# ic(x1_train.shape, x1_test.shape, y_train.shape)   # x1_train.shape: (70, 3), x1_test.shape: (30, 3), y_train.shape: (70,)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3)  #히든레이어

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)  #히든레이어

from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2])   #concatenate(메소드 사용)  #연결만 한거임  
merge1 = Concatenate()([output1, output2])   #Concatenate(클래스 사용)  # 2개니까 리스트 사용
merge2 = Dense(10, name='merge2')(merge1)
merge3 = Dense(5,activation='relu', name='merge3')(merge2)
last_output = Dense(1)(merge3)   #최종 output

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=1)

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test)
ic(results[0])   #mse
ic(results[1])   #metrics['mae]
