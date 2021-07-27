from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from icecream import ic
import numpy as np

### x 데이터 전처리 : x 를 max로 나누기

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# minmaxscalar : 정규화(normalization)
ic(np.min(x), np.max(x))   # np.min(x): 0.0, np.max(x): 711.0

# 데이터 전처리
# 부동소수점 사용(정수보다는 소수점끼리의 연산이 더 빠르니까) 후 X100   <= 데이터 전처리(반드시 해야 함)
# x = x/711.
x = x/np.max(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=9)
# ic(x_test)
# ic(y_test)

# ic(x.shape, x_train.shape, x_test.shape)   # x.shape : (506, 13)   input_dim=13
# ic(y.shape, y_train.shape, y_test.shape)   # (506,)      output = 1(벡터가 1개니까)

# ic(datasets.feature_names)
# ic(datasets.DESCR)   # DESCR : 묘사하다

# train_test_split(70%) 사용,       loss 값, r2(0.7정도는 넘기기) 값 출력

#2. 모델
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가, 예측, r2결정계수
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

'''
#5. 결과값
ic| loss: 16.65877342224121
ic| r2: 0.812320991376007
'''

