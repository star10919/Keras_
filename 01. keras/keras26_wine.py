import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from icecream import ic

# 실습 accuracy 0.8 이상 만들 것!!!

# 1. 데이터
datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data   # (178, 13)
y = datasets.target   # (178,)
# ic(x.shape, y.shape)
# ic(y)


# 원핫인코딩
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
ic(y[:5])
# [[1., 0., 0.],
#  [1., 0., 0.],
#  [1., 0., 0.],
#  [1., 0., 0.],
#  [1., 0., 0.]]
ic(y.shape)   # (178, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=9)

# 1-2. 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일(EarlyStopping), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('accuracy :', results[1])

ic(y_test[-5:-1])
y_predict = model.predict(x_test)
ic(y_predict[-5:-1])


'''
loss : 7.728117452643346e-06
accuracy : 1.0
ic| y_test[-5:-1]: array([[1., 0., 0.],
                          [0., 0., 1.],
                          [0., 1., 0.],
                          [0., 1., 0.]], dtype=float32)
ic| y_predict[-5:-1]: array([[1.0000000e+00, 3.8762810e-10, 1.1142023e-08],
                             [1.4212510e-08, 2.6491962e-08, 1.0000000e+00], 
                             [1.6345615e-12, 1.0000000e+00, 1.5769417e-13], 
                             [5.8870534e-03, 9.9395543e-01, 1.5749357e-04]]
'''