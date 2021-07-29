# 훈련(train)데이터를 10만개로 증폭할것
# 완료 후 기본 모델과 비교
# save_dir도 temp에 넣을것

### 랜덤 ->  [4차원 -> flow] -> concatenate

import numpy as np
from icecream import ic

### 데이터 로드하기

x_train_cifar10 = np.load('./_save/_npy/k55_x_train_cifar10.npy')
x_test_cifar10 = np.load('./_save/_npy/k55_x_test_cifar10.npy')
y_train_cifar10 = np.load('./_save/_npy/k55_y_train_cifar10.npy')
y_test_cifar10 = np.load('./_save/_npy/k55_y_test_cifar10.npy')

# ic(x_train_cifar10)
# ic(x_test_cifar10)
# ic(y_train_cifar10)
# ic(y_test_cifar10)
# ic(x_train_cifar10.shape, x_test_cifar10.shape, y_train_cifar10.shape, y_test_cifar10.shape)

'''
    x_train_cifar10.shape: (50000, 32, 32, 3)
    x_test_cifar10.shape: (10000, 32, 32, 3)
    y_train_cifar10.shape: (50000, 1)
    y_test_cifar10.shape: (10000, 1)
'''

# np.save('./_save/_npy/k55_x_train_cifar10.npy', arr=x_train_cifar10)
# np.save('./_save/_npy/k55_x_test_cifar10.npy', arr=x_test_cifar10)
# np.save('./_save/_npy/k55_y_train_cifar10.npy', arr=y_train_cifar10)
# np.save('./_save/_npy/k55_y_test_cifar10.npy', arr=y_test_cifar10)


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
augment_size=50000

randidx = np.random.randint(x_train_cifar10.shape[0], size=augment_size)
x_augment = x_train_cifar10[randidx].copy()
y_augment = y_train_cifar10[randidx].copy()
ic(x_augment.shape, y_augment.shape)        # (50000, 32, 32, 3), (50000, 1)


#####4차원
# x_augment = x_augment.reshape(x_augment.shape[0], 32, 32, 3)
# x_train = x_train_cifar10.reshape(x_train_cifar10.shape[0], 32, 32, 3)
# x_test = x_test_cifar10.reshape(x_test_cifar10.shape[0], 32, 32, 3)

#####flow
x_augment = train_datagen.flow(# x와 y를 각각 불러옴
            x_augment,  # x
            np.zeros(augment_size),  # y
            batch_size=augment_size,
            save_to_dir='d:/temp/',
            shuffle=False).next()[0]
# ic(type(x_augment), x_augment.shape)       # <class 'numpy.ndarray'>, (40000, 28, 28, 1)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
ic(x_train_cifar10.shape, x_augment.shape)  #(50000, 32, 32, 3), (50000, 32, 32, 3)
ic(y_train_cifar10.shape, y_augment.shape)  #(50000, 1), (50000, 1)

#####concatenate
x_train = np.concatenate((x_train_cifar10, x_augment))
y_train = np.concatenate((y_train_cifar10, y_augment))
ic(x_train.shape, y_train.shape)        #  (100000, 32, 32, 1), (100000, 1)


# ic(np.unique(y_train_cifar10))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] - 10개
# y_train = y_train_cifar10.reshape(-1,1)
# y_test = y_test_cifar10.reshape(-1,1)

# # 1-2. 데이터전처리
# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()
# one.fit(y_train)
# y_train = one.transform(y_train).toarray()
# y_test = one.transform(y_test).toarray()


# 2. 모델구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LSTM, Conv1D

model = Sequential()
#dnn
model.add(Conv2D(128, (2,2), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()


# 3. 컴파일(ES), 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_8_MCP.hdf5')

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=200, validation_split=0.012, callbacks=[es, cp])
end = time.time() - start


# 4. 평가, 예측             predict할 필요는 없다
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test_cifar10, y_test_cifar10)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
# print('loss : ',loss[-10])
print('val_loss : ',val_loss[-10])

'''
*cnn
category : 1.1070876121520996
accuracy : 0.6870999932289124

*dnn
걸린시간 : 80.10825681686401
category : 1.4908812046051025
accuracy : 0.49619999527931213

*LSTM
걸린시간 : 189.16774201393127
category : 2.3025870323181152
accuracy : 0.10000000149011612

*LSTM + Conv1D
걸린시간 : 185.34178686141968
category : 2.3025894165039062
accuracy : 0.10000000149011612

*save model
걸린시간 : 85.17071604728699
category : 2.302586555480957
accuracy : 0.10010000318288803

*checkpoint
category : 2.2929153442382812
accuracy : 0.10400000214576721

*load_npy
걸린시간 : 172.0267460346222
category : 2.052140235900879
accuracy : 0.22370000183582306
'''