from tensorflow.keras.datasets import cifar100
import numpy as np
from icecream import ic

# 1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)
ic(np.max(x_train), np.max(x_test))  # 255

# 1-2. x 데이터 전처리
x_train = x_train/255.
x_test = x_test/255.

# 1-3. y 데이터 전처리
ic(np.unique(y_train))   # 100개
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()   # (50000, 100)
y_test = one.transform(y_test).toarray()     # (10000, 100)
# ic(y_train.shape, y_test.shape)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(102, kernel_size=(2,2), padding='same', input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
# model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()


# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=300, validation_split=0.0025, callbacks=[es])


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('category :', results[0])
print('accuracy :', results[1])


'''
epoch 1000/ batch 600/ patience 20/ val_spl 0.0025
category : 2.7212178707122803
accuracy : 0.3944999873638153

epoch 20/ batch 600/ patience 2/ val_spl 0.0025
category : 2.6021106243133545
accuracy : 0.40959998965263367

model 수정
category : 2.4821901321411133
accuracy : 0.39809998869895935
'''