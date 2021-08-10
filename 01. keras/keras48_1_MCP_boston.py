from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic
from tensorflow.python.keras.layers.core import Dropout

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
# ic(x_test)
# ic(y_test)

# ic(x.shape, x_train.shape, x_test.shape)   # x.shape: (506, 13), x_train.shape: (404, 13), x_test.shape: (102, 13)
# ic(y.shape, y_train.shape, y_test.shape)   # y.shape: (506,), y_train.shape: (404,), y_test.shape: (102,)

#1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler, PowerTransformer
scaler = PowerTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ic(x_train.shape, x_test.shape)   # x_train.shape: (354, 13), x_test.shape: (152, 13)
x_train = x_train.reshape(354, 13, 1)
x_test = x_test.reshape(152, 13, 1)



#2. 모델
# model = Sequential()
# model.add(LSTM(10, input_shape=(13,1), return_sequences=True, activation='relu'))
# model.add(Conv1D(128, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))


#3. 컴파일(ES), 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath='./_save/ModelCheckPoint/keras48_1_MCP.hdf5')

# import time
# start = time.time()
# model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=30, callbacks=[es, cp])
# end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_1_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_1_model_save.h5')   # save model
model = load_model('./_save/ModelCheckPoint/keras48_1_MCP.hdf5')        # checkpoint

#4. 평가, 예측, r2결정계수
loss = model.evaluate(x_test, y_test)
# print("걸린시간 :", end)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)


'''
* PowerTransformer
ic| loss: 5.507851600646973
ic| r2: 0.934103159310994

*cnn + Flatten
ic| loss: 12.241085052490234
ic| r2: 0.8518335236309844

*cnn + GlobalAveragePooling
ic| loss: 65.37923431396484
ic| r2: 0.20864773914491164

*LSTM
걸린시간 : 18.010233879089355
ic| loss: 19.35918426513672
ic| r2: 0.7656758336795892

*Conv1d
걸린시간 : 3.5167489051818848
ic| loss: 15.395515441894531
ic| r2: 0.8136521861168895

*LSTM + Conv1D
걸린시간 : 30.3638174533844
ic| loss: 11.290112495422363
ic| r2: 0.863344116663746

*save model
걸린시간 : 22.779202222824097
ic| loss: 13.407032012939453
ic| r2: 0.8377208619546939

*checkpoint
ic| loss: 11.438616752624512
ic| r2: 0.8615466164910492
'''

