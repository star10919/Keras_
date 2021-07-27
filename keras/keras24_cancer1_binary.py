import numpy as np
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 2진 분류

# 1. 데이터
datasets = load_breast_cancer()
# ic(datasets.DESCR)
# ic(datasets.feature_names)

x = datasets.data
y = datasets.target  #2진 분류(0 or 1)

# ic(x.shape, y.shape)   #(569, 30), (569,)
# ic(y[:40])  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# ic(np.unique(y))   # [0, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9)

# 1-2. 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
model = Sequential()
model.add(Dense(120, input_shape=(30,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # sigmoid : 0과 1사이의 값  # 스칼라 569인 벡터 1개


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # binary_crossentropy : 2진 분류   # metrics(결과에 반영X):평가지표

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min')

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[es])


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)   # 무조건 낮을수록 좋다
print('loss :', results[0])
print('accuracy :', results[1])

ic(y_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])
ic(y_predict)   # sigmoid 통과한 값

# 그래프 그리기
plt.rcParams['font.family'] = 'gulim'
plt.plot(hist.history['loss'])

plt.title('유방암')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend('유방암 loss')
plt.show()


'''
ic| loss: 0.013418859802186489


loss : 0.41947320103645325
accuracy : 0.9766082167625427
'''