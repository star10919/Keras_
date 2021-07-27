import numpy as np
from icecream import ic

### 데이터 로드하기

x_data_diabet = np.load('./_save/_npy/k55_x_data_diabet.npy')
y_data_diabet = np.load('./_save/_npy/k55_y_data_diabet.npy')

ic(x_data_diabet)
ic(y_data_diabet)
ic(x_data_diabet.shape, y_data_diabet.shape)        # x_data_diabet.shape: (442, 10), y_data_diabet.shape: (442,)

# np.save('./_save/_npy/k55_x_data_diabet.npy', arr=x_data_diabet)
# np.save('./_save/_npy/k55_y_data_diabet.npy', arr=y_data_diabet)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data_diabet, y_data_diabet, train_size=0.8, shuffle=True, random_state=9)

# x 데이터 전처리
from sklearn.preprocessing import StandardScaler, PowerTransformer, MaxAbsScaler
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   # x_train.shape: (353, 10), x_test.shape: (89, 10)

x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)

#2. 모델구성(validation)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Flatten
model = Sequential()
model.add(LSTM(10, input_shape=(10, 1), activation='relu', return_sequences=True))
model.add(Conv1D(100, 2, activation='relu'))  #relu : 음수값은 0, 양수만 제대로 잡음
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


#3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_2_MCP.hdf5')

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.1, shuffle=True, verbose=1, callbacks=[es, cp])
end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_2_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_2_model_save.h5')   # save model
# model = load_model('./_save/ModelCheckPoint/keras48_2_MCP.hdf5')        # checkpoint

#4. 평가, 예측(mse, r2)
loss = model.evaluate(x_test, y_test)
# print('걸린시간 :', end)
ic(loss)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
ic(r2)



'''
* MaxAbsScaler
ic| loss: 2240.12841796875
ic| r2: 0.588361118619634

*cnn + Flatten
ic| loss: 2341.884765625
ic| r2: 0.5696626833347622

*cnn + GAP
ic| loss: 3888.330810546875
ic| r2: 0.2854926965183139

*LSTM
걸린시간 : 88.2120795249939
ic| loss: 2421.381591796875
ic| r2: 0.5550545863459289

*Conv1D
걸린시간 : 16.57935118675232
ic| loss: 2188.756591796875
ic| r2: 0.5978010110126091

*LSTM + Conv1D
걸린시간 : 91.52887654304504
ic| loss: 2080.7890625
ic| r2: 0.6176408093415701

*save model
걸린시간 : 43.92455267906189
ic| loss: 2508.6279296875
ic| r2: 0.5390224883761505

*checkpoint
ic| loss: 2179.181884765625
ic| r2: 0.5995604279480531

*load_npy
ic| loss: 2319.772705078125
ic| r2: 0.5737259521590647
'''