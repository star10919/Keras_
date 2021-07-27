import numpy as np
from icecream import ic
from sklearn.model_selection import train_test_split

### 데이터 로드하기

x_data_cancer = np.load('./_save/_npy/k55_x_data_cancer.npy')
y_data_cancer = np.load('./_save/_npy/k55_y_data_cancer.npy')

ic(x_data_cancer)
ic(y_data_cancer)
ic(x_data_cancer.shape, y_data_cancer.shape)        # x_data_cancer.shape: (569, 30), y_data_cancer.shape: (569,)

# np.save('./_save/_npy/k55_x_data_cancer.npy', arr=x_data_cancer)
# np.save('./_save/_npy/k55_y_data_cancer.npy', arr=y_data_cancer)


x_train, x_test, y_train, y_test = train_test_split(x_data_cancer, y_data_cancer, train_size=0.7, random_state=9)

# 1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   # x_train.shape: (398, 30), x_test.shape: (171, 30)
x_train = x_train.reshape(398, 30, 1)
x_test = x_test.reshape(171, 30, 1)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Conv1D,Flatten,Dense
model = Sequential()
model.add(LSTM(128, input_shape=(30,1), activation='relu', return_sequences=True))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # sigmoid : 0과 1사이의 값  # 스칼라 569인 벡터 1개


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # binary_crossentropy : 2진 분류   # metrics(결과에 반영X):평가지표

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=5, mode='min')
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', filepath='./_save/ModelCheckPoint/keras48_3_MCP.hdf5')

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, validation_split=0.02, batch_size=100, callbacks=[es, cp])
end = time.time() - start

model.save('./_save/ModelCheckPoint/keras48_3_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_3_model_save.h5')   # save model
# model = load_model('./_save/ModelCheckPoint/keras48_3_MCP.hdf5')        # checkpoint


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)   # 무조건 낮을수록 좋다
# print('걸린시간 :', end)
print('loss :', results[0])
print('accuracy :', results[1])

# ic(y_test[-5:-1])
# y_predict = model.predict(x_test[-5:-1])
# ic(y_predict)  # sigmoid 통과한 값

# 그래프 그리기
# plt.rcParams['font.family'] = 'gulim'
# plt.plot(hist.history['loss'])

# plt.title('유방암')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend('유방암 loss')
# plt.show()


'''
loss : 0.41947320103645325
accuracy : 0.9766082167625427

*cnn + Flatten
걸린시간 : 55.207491874694824
loss : 0.0763908177614212
accuracy : 0.9707602262496948

*cnn + GAP
걸린시간 : 34.21939539909363
loss : 0.2451203167438507
accuracy : 0.9122806787490845

*LSTM
걸린시간 : 9.949499607086182
loss : 0.18049414455890656
accuracy : 0.9356725215911865

*LSTM + Conv1D
걸린시간 : 24.275282382965088
loss : 0.03207003325223923
accuracy : 0.9941520690917969

*save model
걸린시간 : 20.550546646118164
loss : 0.06655818223953247
accuracy : 0.9707602262496948

*checkpoint
loss : 0.06175762787461281
accuracy : 0.9707602262496948

*load_npy
loss : 0.05980822816491127
accuracy : 0.9707602262496948
'''