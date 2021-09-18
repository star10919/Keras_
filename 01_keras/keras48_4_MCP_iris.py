import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, LSTM, Conv1D

### 다중 분류 (0,1,2를 명목척도로 바꿔줘야 함 <- 원핫인코딩 사용(to_categorical))
### Output activation = 'softmax', 컴파일 loss = 'categorical_crossentropy'

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)


# 원핫인코딩 One-Hot-Encoding  (여자는 남자의 2배가 아니므로 원핫인코딩을 해줘야 함)
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

# [0, 1, 2, 1] ->
# [[1, 0, 0]
#  [0, 1, 0]
#  [0, 0, 1]
#  [0, 1, 0]]   (4,) -> (4, 3) (데이터 수, 데이터 종류(0,1,2))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9)

# 1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ic(x_train.shape, x_test.shape)   #  x_train.shape: (105, 4), x_test.shape: (45, 4)
x_train = x_train.reshape(105, 4, 1)
x_test = x_test.reshape(45, 4, 1)

# 1-3. y 데이터 전처리
from tensorflow.keras.utils import to_categorical    # 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
ic(y[:5])
# [0,0,0,0,0]
# [[1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]]
ic(y.shape)   # (150, 3)


# 2. 모델
# model = Sequential()
# model.add(LSTM(128, activation='relu', input_shape=(4,1), return_sequences=True))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))  # softmax : 다중분류   # 0,1,2  3개라서 3개로 나와야 함(150, 3)


# 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # binary_crossentropy : 2진 분류   # metrics(결과에 반영X):평가지표

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_4_MCP.hdf5')

# import time
# start = time.time()
# model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es, cp])
# end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_4_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_4_model_save.h5')   # save model
model = load_model('./_save/ModelCheckPoint/keras48_4_MCP.hdf5')        # checkpoint

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)   # 무조건 낮을수록 좋다
# print('걸린시간 :', end)
print('loss :', results[0])
print('accuracy :', results[1])

ic(y_test[:5])
y_predict = model.predict(x_test[:5])
ic(y_predict)   # 소프트맥스 통과한 값


'''
loss : 0.03401293233036995
accuracy : 1.0

*cnn + Flatten
걸린시간 : 3.87078595161438
loss : 0.022757794708013535
accuracy : 1.0

*cnn + GAP
걸린시간 : 3.823838233947754
loss : 0.13620592653751373
accuracy : 0.9555555582046509

*LSTM
걸린시간 : 6.369047403335571
loss : 0.06772168725728989
accuracy : 1.0

*LSTM + Conv1D
걸린시간 : 5.24717903137207
loss : 0.045467764139175415
accuracy : 1.0

*save model
걸린시간 : 8.921130418777466
loss : 0.021816492080688477
accuracy : 1.0

*checkpoint
loss : 0.01563490554690361
accuracy : 1.0
'''