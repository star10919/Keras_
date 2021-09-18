from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from icecream import ic

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)
# ic(x_test)
# ic(y_test)

ic(x.shape, x_train.shape, x_test.shape)   # x.shape : (506, 13)   input_dim=13
ic(y.shape, y_train.shape, y_test.shape)   # (506,)      output = 1(벡터가 1개니까)

# ic(datasets.feature_names)
# ic(datasets.DESCR)   # DESCR : 묘사하다

# train_test_split(70%) 사용,       loss 값, r2(0.7정도는 넘기기) 값 출력

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=1000, batch_size=1)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.3, shuffle=True)


#4. 평가, 예측, r2결정계수
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

'''
#5. 결과값
ic| loss: 20.64463996887207
ic| r2: 0.7501166183019436
'''

