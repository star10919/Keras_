from tensorflow.keras.datasets import cifar100
import numpy as np
from icecream import ic

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)-1차원
ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)
ic(np.max(x_train), np.max(x_test))  # 255

# 1-2. x 데이터 전처리 - scaler:2차원에서만 가능
x_train = x_train.reshape(50000, 32 * 32 * 3)    #(50000, 3072)
x_test = x_test.reshape(10000, 32 * 32 * 3)      #(10000, 3072)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)   # fit_transform 은  x_train 에서만 사용한다!!!!!!!!!!!!
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)    # 4차원으로 다시 reshape(Conv2d 사용해야 되니까)
x_test = x_test.reshape(10000, 32, 32, 3)

# 1-3. y 데이터 전처리
ic(np.unique(y_train))   # 100개
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder    # sklearn으로 되어 있는 애들은 모두 2차원으로 해줘야 함/OneHotEncoder는 무조건 2차원으로 해줘야 함
one = OneHotEncoder()
# one.fit(y_train)
y_train = one.fit_transform(y_train).toarray()   # (50000, 100)
y_test = one.transform(y_test).toarray()     # (10000, 100)
# ic(y_train.shape, y_test.shape)

# from tensorflow.keras.utils import to_categorical   #0,1,2 값이 없어도 무조건 생성/shape유연
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3))) 
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())        

model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))                   
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())            

model.add(Conv2D(64, (2, 2), activation='relu'))                   
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))

model.add(Flatten())                                              
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()


# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=7, verbose=1)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.25, callbacks=[es])
end_time = time.time() - start_time


# 4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=128)
print("=============================================")
print("걸린 시간 :", end_time)
print('loss :', results[0])
print('acc :', results[1])

#plt 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))   # figure : 판 깔겠다.

# 1
plt.subplot(2,1,1)   # subplot : 그림 2개 그리겠다.
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()



'''
걸린 시간 : 50.16567611694336
category : 2.7187631130218506
accuracy : 0.4185999929904938
'''