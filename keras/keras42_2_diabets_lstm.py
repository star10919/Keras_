from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# 실습 MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer 각 결과 적어놓기

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # (442, 10)  (442,)

# ic(datasets.feature_names)   
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

# ic(x[:30])
# ic(np.min(y), np.max(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

# x 데이터 전처리
from sklearn.preprocessing import StandardScaler, PowerTransformer, MaxAbsScaler
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   # x_train.shape: (353, 10), x_test.shape: (89, 10)

x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)

#2. 모델구성(validation)
model = Sequential()
model.add(LSTM(100, input_shape=(10,1), activation='relu'))  #relu : 음수값은 0, 양수만 제대로 잡음
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


#3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.1, shuffle=True, verbose=1)
end = time.time() - start

#4. 평가, 예측(mse, r2)
loss = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
ic(loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
ic(r2)



'''
* MaxAbsScaler
ic| loss: 2240.12841796875
ic| r2: 0.588361118619634

*cnn + Flatten
ic| loss: 2341.884765625
ic| r2: 0.5696626833347622

*cnn + GAP
ic| loss: 3888.330810546875
ic| r2: 0.2854926965183139

*LSTM
걸린시간 : 88.2120795249939
ic| loss: 2421.381591796875
ic| r2: 0.5550545863459289
'''