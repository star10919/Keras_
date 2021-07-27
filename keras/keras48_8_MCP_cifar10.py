from tensorflow.keras.datasets import cifar10
import numpy as np
from icecream import ic

# 1. 데이터
(x_train,y_train), (x_test, y_test) = cifar10.load_data()
# ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
# ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)

x_train = x_train.reshape(50000, 32, 96)
x_test = x_test.reshape(10000, 32, 96)

ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] - 10개
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# 1-2. 데이터전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
# ic(y_train.shape, y_test.shape)   # (50000, 10), (10000, 10)

# 2. 모델구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LSTM, Conv1D

# model = Sequential()
# dnn
# model.add(LSTM(128, activation='relu', input_shape=(32, 96), return_sequences=True))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()


# 3. 컴파일(ES), 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_8_MCP.hdf5')

# import time
# start = time.time()
# model.fit(x_train, y_train, epochs=1000, batch_size=512, validation_split=0.012, callbacks=[es, cp])
# end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_8_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_8_model_save.h5')           # save model
model = load_model('./_save/ModelCheckPoint/keras48_8_MCP.hdf5')                # checkpoint

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])


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
'''