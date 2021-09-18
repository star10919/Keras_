from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt


# 1. 데이터
(x_train, y_train),(x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

ic(x_train[0], type(x_train[0]))
ic(y_train[0])

ic(len(x_train[0]), len(x_train[11]))  #  len(x_train[0]): 87, len(x_train[11]): 59    # 길이가 다르니까 앞에서부터 패딩해줘야 함

# ic(x_train[0].shape)        # AttributeError: 'list' object has no attribute 'shape'

ic(x_train.shape, x_test.shape)     # x_train.shape: (8982,), x_test.shape: (2246,)
ic(y_train.shape, y_test.shape)     # y_train.shape: (8982,), y_test.shape: (2246,)

ic(type(x_train))       # type(x_train): <class 'numpy.ndarray'>

print('뉴스기사의 최대길이 :', max(len(i) for i in x_train))    # 뉴스기사의 최대길이 : 2376
# print('뉴스기사의 최대길이 :', max(len(x_train)))           # 이건 안 됨!!!
print('뉴스기사의 평균길이 :', sum(map(len, x_train)) / len(x_train))     # 뉴스기사의 평균길이 : 145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
ic(x_train.shape, x_test.shape)     # x_train.shape: (8982, 100), x_test.shape: (2246, 100)
ic(type(x_train), type(x_train[0]))     #  type(x_train): <class 'numpy.ndarray'>,   type(x_train[0]): <class 'numpy.ndarray'>
ic(x_train[0])          # 100-87 =13개의 0이 앞에 붙음

# y 확인
ic(np.unique(y_train))       # 0-45  총 46개  ->  softmax, categorical crossentropy, 평가지표 : acc

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
ic(y_train.shape, y_test.shape)     # y_train.shape: (8982, 46), y_test.shape: (2246, 46)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=77, input_length=100))
model.add(LSTM(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode=min, verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split=0.2)


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('category :', results[0])
print('acc :', results[1])