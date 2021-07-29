# 훈련(train)데이터를 10만개로 증폭할것
# 완료 후 기본 모델과 비교
# save_dir도 temp에 넣을것

### 랜덤 ->  [4차원 -> flow] -> concatenate

import numpy as np
from icecream import ic

### 데이터 로드하기

# (x_train_minst, y_train_mnist), (x_test_minst, y_test_minst) = mnist.load_data()
# np.save('./_save/_npy/k55_x_train_mnist.npy', arr=x_train_minst)
# np.save('./_save/_npy/k55_x_test_mnist.npy', arr=x_test_minst)
# np.save('./_save/_npy/k55_y_train_mnist.npy', arr=y_train_mnist)
# np.save('./_save/_npy/k55_y_test_mnist.npy', arr=y_test_minst)

x_train_mnist = np.load('./_save/_npy/k55_x_train_mnist.npy')
x_test_mnist = np.load('./_save/_npy/k55_x_test_mnist.npy')
y_train_mnist = np.load('./_save/_npy/k55_y_train_mnist.npy')
y_test_mnist = np.load('./_save/_npy/k55_y_test_mnist.npy')

ic(x_train_mnist)
ic(x_test_mnist)
ic(y_train_mnist)
ic(y_test_mnist)
ic(x_train_mnist.shape, x_test_mnist.shape, y_train_mnist.shape, y_test_mnist.shape)

'''
    x_train_mnist.shape: (60000, 28, 28)
    x_test_mnist.shape: (10000, 28, 28)
    y_train_mnist.shape: (60000,)
    y_test_mnist.shape: (10000,)
'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

## ImageDataGenerator로 데이터 증폭시키기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest',
    )

#####랜덤
# 데이터 증폭
augment_size=40000

randidx = np.random.randint(x_train_mnist.shape[0], size=augment_size)
x_augment = x_train_mnist[randidx].copy()
y_augment = y_train_mnist[randidx].copy()
ic(x_augment.shape, y_augment.shape)        # (40000, 28, 28), (40000,)


#####4차원
x_augment = x_augment.reshape(x_augment.shape[0], 28, 28, 1)
x_train = x_train_mnist.reshape(x_train_mnist.shape[0], 28, 28, 1)
x_test = x_test_mnist.reshape(x_test_mnist.shape[0], 28, 28, 1)

#####flow
x_augment = train_datagen.flow(# x와 y를 각각 불러옴
            x_augment,  # x
            np.zeros(augment_size),  # y
            batch_size=augment_size,
            save_to_dir='d:/temp/',
            shuffle=False).next()[0]
# ic(type(x_augment), x_augment.shape)       # <class 'numpy.ndarray'>, (40000, 28, 28, 1)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
ic(x_train.shape, x_augment.shape)
ic(y_train_mnist.shape, y_augment.shape)

#####concatenate
x_train = np.concatenate((x_train, x_augment))
y_train = np.concatenate((y_train_mnist, y_augment))
ic(x_train.shape, y_train.shape)        #  (100000, 28, 28, 1), (100000,)


ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  : 10개
# y_train = y_train.reshape(-1,1)
# y_test = y_test_mnist.reshape(-1,1)

# # 1-2. x 데이터 전처리
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 1-3. y 데이터 전처리
# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()
# one.fit(y_train)
# y_train = one.transform(y_train).toarray() # (60000, 10)
# y_test = one.transform(y_test).toarray() # (10000, 10)
# # ic(y_train.shape, y_test.shape)


# 2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Dropout, GlobalAveragePooling1D, LSTM, Conv1D

model = Sequential()
# dnn
# model.add(LSTM(100, input_shape=(28, 28), return_sequences=True))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# cnn
model.add(Conv2D(filters=100, kernel_size=(4,4), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))
model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
model.add(Conv2D(32, (2, 2), padding='valid', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()


# 3. 컴파일, 훈련           metrics['acc]
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_6_MCP.hdf5')


import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=512, validation_split=0.0012, callbacks=[es, cp])
end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_6_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_6_model_save.h5')       # save model
model = load_model('./_save/ModelCheckPoint/keras48_6_MCP.hdf5')            # checkpoint

# 4. 평가, 예측             predict할 필요는 없다
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test_mnist)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
# print('loss : ',loss[-10])
print('val_loss : ',val_loss[-10])


'''
category : 0.04307371377944946
accruacy : 0.9900000095367432

*dnn
걸린시간 : 67.98393535614014
category : 0.10037190467119217

*cnn + GlobalAveragePooling
category : 0.06292704492807388
accruacy : 0.9890999794006348

*LSTM
걸린시간 : 30.924014568328857
category : 0.22128017246723175
accruacy : 0.9373999834060669

*LSTM + Conv1D
걸린시간 : 26.39584970474243
category : 0.0646396353840828
accruacy : 0.9832000136375427

*save model
걸린시간 : 35.15260910987854
category : 0.06562792509794235
accruacy : 0.9840999841690063

*checkpoint
category : 0.0692889615893364
accruacy : 0.9833999872207642

*load_npy
걸린시간 : 31.242682218551636
category : 0.07776498794555664
accruacy : 0.9807000160217285

*flow
Epoch 00047: early stopping
313/313 [==============================] - 1s 4ms/step - loss: 0.0546 - acc: 0.9861 
acc :  0.9750900864601135
val_acc :  0.9583333134651184
val_loss :  0.11733803898096085
'''