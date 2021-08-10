import numpy as np
from icecream import ic

### 함수형 모델

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)])
x = np.transpose(x)
# ic(x.shape)   # (100, 5)   # input_shape=(5,)
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
# ic(y.shape)   # (100, 2)   # oupput=(2,)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model   # Sequential: 순차형 모델, Model: 함수형 모델
from tensorflow.keras.layers import Dense, Input

# 함수형은 레이어마다 변수명을 정해줘야 함.  /  함수형은 모델에 대한 정의를 맨 마지막에 해줌  /  장점 : 모델을 합칠 수 있음(시작지점과 끝지점을 지정함으로써), 융통성이 있음
input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()



# model = Sequential()   /   sequential은  모델에 대한 정의를 맨 처음에 해줌  /  단일모델일 때는 시퀀셜이 더 편함
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

# model.summary()


#3. 컴파일, 훈련


#4. 평가, 예측
