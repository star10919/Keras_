import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

### activation

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

ic(x.shape, y.shape)  # (442, 10)  (442,)

ic(datasets.feature_names)   
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
ic(datasets.DESCR)

ic(x[:30])
ic(np.min(y), np.max(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성(validation)
model = Sequential()
model.add(Dense(70, input_shape=(10,), activation='relu'))  # 활성화함수 (default값 존재)
model.add(Dense(30, activation='relu'))  # relu : 음수값은 0, 양수만 제대로 잡음(relu 쓰면 성능 좋아짐)
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='sigmoid'))  # sigmoid : 0~1까지의 값으로 한정시켜서 숫자가 막대히 커지는 것을 막음
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))   # 마지막행에 고정되어 있는 activation이 있어서 활성화함수 쓰면 안됨.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=40, validation_split=0.3, shuffle=True)

#4. 평가, 예측(mse, r2)
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
ic(r2)