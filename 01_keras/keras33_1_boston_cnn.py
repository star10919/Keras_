from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic
from tensorflow.python.keras.layers.core import Dropout

# 실습 MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer 각 결과 적어놓기

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
# ic(x_test)
# ic(y_test)

# ic(x.shape, x_train.shape, x_test.shape)   # x.shape: (506, 13), x_train.shape: (404, 13), x_test.shape: (102, 13)
# ic(y.shape, y_train.shape, y_test.shape)   # y.shape: (506,), y_train.shape: (404,), y_test.shape: (102,)

#1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler, PowerTransformer
scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ic(x_train.shape, x_test.shape)   # x_train.shape: (354, 13), x_test.shape: (152, 13)
x_train = x_train.reshape(354, 13, 1, 1)
x_test = x_test.reshape(152, 13, 1, 1)



#2. 모델
model = Sequential()
model.add(Conv2D(128, kernel_size=(1,1), padding='valid', input_shape=(13, 1, 1), activation='relu'))
model.add(Conv2D(128, (1,1), padding='same', activation='relu'))
model.add(Conv2D(128, (1,1), padding='valid', activation='relu'))
model.add(Conv2D(64, (1,1), padding='valid', activation='relu'))
model.add(Conv2D(64, (1,1), padding='valid', activation='relu'))
model.add(Conv2D(32, (1,1), padding='valid', activation='relu'))
model.add(Conv2D(16, (1,1), padding='valid', activation='relu'))
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(1))


#3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=30, callbacks=[es])
end = time.time() - start


#4. 평가, 예측, r2결정계수
loss = model.evaluate(x_test, y_test)
print("걸린시간 :", end)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)


'''
* MaxAbsScaler
ic| loss: 7.193202018737793
ic| r2: 0.9139393522489021

* RobustScaler
ic| loss: 7.381626605987549
ic| r2: 0.9116850132736721

* QuantileTransformer
ic| loss: 5.611426830291748
ic| r2: 0.9328639728985081

* PowerTransformer
ic| loss: 5.507851600646973
ic| r2: 0.934103159310994

#cnn + Flatten
ic| loss: 12.241085052490234
ic| r2: 0.8518335236309844

#cnn + GlobalAveragePooling
ic| loss: 65.37923431396484
ic| r2: 0.20864773914491164
'''

