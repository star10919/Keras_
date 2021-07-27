import numpy as np
from tensorflow.keras.datasets import cifar100, mnist
from icecream import ic

### 가중치 저장(아주 중요!!!!!!!) / 순수하게 모델만 저장(나머지 다 주석처리) - 확장자는 무조건 .h5

'''
# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

ic(x_train.shape, y_train.shape)   
ic(x_test.shape, y_test.shape)     

# 1-2. x 데이터 전처리 - scaler:2차원에서만 가능
x_train = x_train.reshape(60000, 28 * 28)   # 4차원 -> 2차원
x_test = x_test.reshape(10000, 28 * 28)
# 전처리 하기 -> scailing
print(x_train.shape, x_test.shape) # (50000, 3072)-2차원, (10000, 3072)-2차원

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # fit_transform 은  x_train 에서만 사용한다!!!!!!!!!!!!
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)    # 4차원으로 다시 reshape(Conv2d 사용해야 되니까)
x_test = x_test.reshape(10000, 28, 28, 1)

# 1-3. y 데이터 전처리 -> one-hot-encoding
ic(np.unique(y_train))    # 100개
# from tensorflow.keras.utils import to_categorical   # 0,1,2 값이 없어도 무조건 생성/shape유연
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)

from sklearn.preprocessing import OneHotEncoder    # sklearn으로 되어 있는 애들은 모두 2차원으로 해줘야 함/OneHotEncoder는 무조건 2차원으로 해줘야 함
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

one = OneHotEncoder()
# one.fit(y_train)
y_train = one.fit_transform(y_train).toarray()   # (50000, 100)
y_test = one.transform(y_test).toarray()     # (10000, 100)
# ic(y_train.shape, y_test.shape)
'''

# 2. 모델 구성(GlobalAveragePooling2D 사용)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
model = Sequential()
model.add(Conv2D(128, kernel_size=(2, 2), 
                    padding='valid', input_shape=(28, 28, 1), activation='relu'))
# model.add(Dropout(0, 2)) # 20% node Dropout
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))   
model.add(MaxPool2D()) 
model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D()) 
model.add(Conv2D(64, (2,2), padding='valid', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D()) 
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.save('./_save/keras45_1_save_model.h5')  # 모델 저장        # 저장되는 확장자 : h5         # ./ : 현재위치(STUDY 폴더)

'''
# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

import time 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, verbose=1, callbacks=[es], validation_split=0.2, shuffle=True, batch_size=100)
end_time = time.time() - start_time


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("걸린시간 :", end_time)
print('category :', results[0])
print('accuracy :', results[1])


# # 시각화 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# # 1
# plt.subplot(2, 1, 1) # 2개의 플롯을 할건데, 1행 1열을 사용하겠다는 의미 
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# # 2
# plt.subplot(2, 1, 2) # 2개의 플롯을 할건데, 1행 2열을 사용하겠다는 의미 
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()



걸린시간 : 98.12368416786194
category : 0.03186216577887535
accuracy : 0.991599977016449
'''