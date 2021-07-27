# 실습 diabets
# 1. loss와 R2로 평가 함
# Minmax와 Standard 결과 명시

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

# 데이터 전처리
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성(validation)
model = Sequential()
model.add(Dense(100, input_shape=(10,), activation='relu'))  #relu : 음수값은 0, 양수만 제대로 잡음
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.1, shuffle=True, verbose=1)

#4. 평가, 예측(mse, r2)
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
ic(r2)



'''
* MinmaxcScaler
ic| loss: 1916.69140625
ic| r2: 0.6477948564610096

* StandardScaler
ic| loss: 3659.23486328125
ic| r2: 0.3275906148072223
'''