from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from icecream import ic
import numpy as np
# MinmaxScaler : 정규분포에 맞춰서 scaler 함

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# ic(np.min(x), np.max(x))   # np.min(x): 0.0, np.max(x): 711.0

# 데이터 전처리
# 부동소수점 사용(정수보다는 소수점끼리의 연산이 더 빠르니까) 후 X100   <= 데이터 전처리(반드시 해야 함)
# (방법 1)x = x/711.
# (방법 2)x = x/np.max(x)
# (방법 3)x = (x - np.min(x)) / (np.max(x) - np.min(x))     # x = (x - min) / (max - min)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=9)
# ic(x_test)
# ic(y_test)

# (방법 4)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)   ##1- 실행만 시킨거임   / train만을 fit 시켜서(전체 데이터로 minmaxscaler하면 과적합 됨)
x_train = scaler.transform(x_train)   ##2- 변환(scaler됨)   / train 기준에 스케일된 걸로 test transfrom해줌
x_test = scaler.transform(x_test)
# x_pred = scaler.transform(x_pred)
   #  =>test 데이터는 train 데이터에 반영되면 안된다!!!!!!!!!!!!!!!!!!!



ic(x.shape, x_train.shape, x_test.shape)   # x.shape : (506, 13)   input_dim=13
ic(y.shape, y_train.shape, y_test.shape)   # (506,)      output = 1(벡터가 1개니까)

# train_test_split(70%) 사용,       loss 값, r2(0.7정도는 넘기기) 값 출력

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=20)


#4. 평가, 예측, r2결정계수
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

'''
#5. 결과값
ic| loss: 7.853621482849121
ic| r2: 0.9115205049720955
'''


