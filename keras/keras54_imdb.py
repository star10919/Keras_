from tensorflow.keras.datasets import imdb
from icecream import ic
import numpy as np

### 실습

# 1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

ic(x_train[0], type(x_train[0]))        # <class 'list'>
ic(y_train[0])      # 1

ic(len(x_train[0]), len(x_train[11]))       # len(x_train[0]): 218, len(x_train[11]): 99

ic(x_train.shape, x_test.shape)     # x_train.shape: (25000,), x_test.shape: (25000,)
ic(y_train.shape, y_test.shape)     # y_train.shape: (25000,), y_test.shape: (25000,)

ic(type(x_train))       # <class 'numpy.ndarray'>

print('최대길이 :', max(len(i) for i in x_train))      #2494
print('평균길이 :', sum(map(len, x_train)) / len(x_train))     #238.714


# 1-2. 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=200, padding='pre')
x_test = pad_sequences(x_test, maxlen=200, padding='pre')
ic(x_train.shape, x_test.shape)     # x_train.shape: (25000, 100), x_test.shape: (25000, 100)
ic(type(x_train), type(x_train[0]))     # <class 'numpy.ndarray'>, <class 'numpy.ndarray'>
ic(x_train[0])

ic(np.unique(y_train))      # array([0, 1]  -> sigmoid, binary_crossentropy, acc
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
ic(y_train.shape, y_test.shape)     # y_train.shape: (25000, 2), y_test.shape: (25000, 2)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, LSTM, Embedding, GRU

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=200, input_length=200))
model.add(GRU(100, activation='relu', return_sequences=True))
model.add(Conv1D(100, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# 3. 컴파일(ES), 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=7, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=13, batch_size=1000, validation_split=0.2, callbacks=[es])
end = time.time() - start


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
print('binary :', results[0])
print('acc :', results[1])


'''
Epoch 00010: early stopping
782/782 [==============================] - 81s 103ms/step - loss: 0.9340 - acc: 0.8401
걸린시간 : 89.68203496932983
binary : 0.9339950084686279
acc : 0.8401200175285339
'''