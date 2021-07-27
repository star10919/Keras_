from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from icecream import ic

# 실습 MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer 각 결과 적어놓기

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)
# ic(x_test)
# ic(y_test)

# ic(x.shape, x_train.shape, x_test.shape)   # x.shape : (506, 13)   input_dim=13
# ic(y.shape, y_train.shape, y_test.shape)   # (506,)      output = 1(벡터가 1개니까)

# ic(datasets.feature_names)
# ic(datasets.DESCR)   # DESCR : 묘사하다

# train_test_split(70%) 사용,       loss 값, r2(0.7정도는 넘기기) 값 출력


# 데이터 전처리
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=30)
# model.fit(x_train, y_train, epochs=1500)


#4. 평가, 예측, r2결정계수
loss = model.evaluate(x_test, y_test)
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
'''

