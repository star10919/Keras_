import numpy as np
from icecream import ic

### 데이터 로드하기

x_data_wine = np.load('./_save/_npy/k55_x_data_wine.npy')
y_data_wine = np.load('./_save/_npy/k55_y_data_wine.npy')

ic(x_data_wine)
ic(y_data_wine)
ic(x_data_wine.shape, y_data_wine.shape)        # x_data_wine.shape: (178, 13), y_data_wine.shape: (178,)

# np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine)
# np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)

y_data_wine = y_data_wine.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder    # 0, 1, 2 자동 채움 안됨 / # to_categorical 0, 1, 2 없으나 자동 생성
onehot = OneHotEncoder()
onehot.fit(y_data_wine)
y = onehot.transform(y_data_wine).toarray() 
ic(y.shape)    # (4898, 7)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data_wine, y, train_size=0.995, shuffle=True, random_state=24)

# x 데이터 전처리(scaler)
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

ic(x_train.shape, x_test.shape)   #  x_train.shape: (4873, 11), x_test.shape: (25, 11)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# 2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Conv1D, LSTM
model = Sequential()
model.add(LSTM(240, activation='relu', input_shape=(13,1), return_sequences=True))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(240, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일(EalryStopping), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_5_MCP.hdf5')

import time
start = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=512, validation_split=0.0024, callbacks=[es, cp])
end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_5_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_5_model_save.h5')   # save model
# model = load_model('./_save/ModelCheckPoint/keras48_5_MCP.hdf5')        # checkpoint

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])

ic(y_test[-5:-1])
y_predict = model.predict(x_test)
ic(y_predict[-5:-1])


'''
category : 0.9682848453521729
accuracy : 0.800000011920929

*cnn + Flatten
걸린시간 : 13.947882652282715
category : 0.9115962386131287
accuracy : 0.6399999856948853

*cnn + GAP
걸린시간 : 29.681384325027466
category : 1.0343053340911865
accuracy : 0.4399999976158142

*LSTM
걸린시간 : 50.68032455444336
category : 0.8400266766548157
accuracy : 0.6000000238418579

*LSTM + Conv1D
걸린시간 : 62.63808012008667
category : 0.7918718457221985
accuracy : 0.6399999856948853

*save model
걸린시간 : 26.757463216781616
category : 0.7706682682037354
accuracy : 0.6000000238418579

*checkpoint
category : 0.7948649525642395
accuracy : 0.6399999856948853

*load_npy
category : 0.0
accuracy : 1.0
'''