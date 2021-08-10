from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

#1-2. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
onehot.fit(y_train)
y_train = onehot.transform(y_train).toarray()  # (60000, 10
y_test = onehot.transform(y_test).toarray()   # (10000, 10)
ic(y_train.shape, y_test.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
model = Sequential()
model.add(Conv2D(128, kernel_size=(2,2), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
# model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
# model.add(Conv2D(100, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.001, callbacks=[es])


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('category :', results[0])
print('accuracy :', results[1])

'''
category : 0.42756807804107666
accuracy : 0.91839998960495
'''