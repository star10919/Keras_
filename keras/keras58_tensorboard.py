import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from icecream import ic

### TensorBoard - 콜백에 있음
# (cmd창에서 로컬호스트 주소 받아서 크롬에서 그래프 사이트 열어서 볼 수 있음)
'''
cmd창에서
d:
cd study
cd _save
cd _graph
dir/w
tensorboard --logdir=.

크롬 키고 주소창에
http://127.0.0.1:6006  or  http://localhost:6006/
'''



# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ic(x_train.shape, y_train.shape)   # x_train.shape: (60000, 28, 28), y_train.shape: (60000,)
# ic(x_test.shape, y_test.shape)     # x_test.shape: (10000, 28, 28), y_test.shape: (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)   # reshape : 내용물과 순서는 바뀌지 않음(차원만 바뀜)  -  총 곱셈이 같아야 함.
x_test = x_test.reshape(10000, 28, 28, 1)   # 컨벌루션은 무조건 4차원으로 만들어서 넣어줘야 함.

ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


# #1-2. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)
# ic(y_train.shape, y_test.shape)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4), padding='same', input_shape=(28, 28, 1)))
# model.add(Conv2D(16, (4,4), padding='same', activation='relu'))
model.add(Conv2D(16, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())   # 연산X
model.add(Flatten())  # shape자체가 2차원이 됨(Dense써야 되니까-Dense:2차원)(연산은 안함)
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


# 3. 컴파일, 훈련           metrics['acc]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0, write_graph=True, write_images=True)

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.025, callbacks=[es, tb])
end = time.time() - start

# 4. 평가, 예측             predict할 필요는 없다
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
print('category :', results[0])
print('accruacy :', results[1])


# acc로만 판단해보자.
# 0.98 이상 완성


'''
category : 0.04307371377944946
accruacy : 0.9900000095367432
'''