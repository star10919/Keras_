import numpy as np
import pandas as pd
from icecream import ic

# 다중분류
# 모델링하고
# 0.8 이상 완성


# 1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',       # 경로잡기 중요!
                        index_col=None, header=0)    #header=0 첫번째라인   # (4898,12)

# * visual studio 기준(파이참이랑 다름)
    # ./  :  현재폴더(STUDY)
    # ../ :  상위폴더

# print(datasets)   # quality 가  y

# 아래 3개는 꼭 찍어보기
    # ic(datasets.shape)   # (4898, 12)   => x:(4898,11),   y:(4898,) 으로 잘라주기
    # ic(datasets.info())
    # ic(datasets.describe())




#1 판다스 -> 넘파이 : index와 header가 날아감
#2 x와 y를 분리
#3. sklearn의 OneHotEncoder 사용할것
#3 y의 라벨을 확인 np.unique(y) <= Output node 개수 잡아주기 위해서
#5. y의 shape 확인 (4898,) -> (4898,7)


datasets_np = datasets.to_numpy()   #1 판다스 -> 넘파이
ic(datasets_np)
x = datasets_np[:,0:11]
ic(x)
y = datasets_np[:,[-1]]
ic(y)
ic(x.shape, y.shape)   # x.shape: (4898, 11), y.shape: (4898,1)
ic(np.unique(y))   # [3, 4, 5, 6, 7, 8, 9]  -  7개

# y 데이터 전처리
# 원핫인코딩
# from tensorflow.keras.utils import to_categorical   # to_categorical 0, 1, 2 없으나 자동 생성
# y = to_categorical(y)
# ic(y)

from sklearn.preprocessing import OneHotEncoder    # 0, 1, 2 자동 채움 안됨 / # to_categorical 0, 1, 2 없으나 자동 생성
onehot = OneHotEncoder()
onehot.fit(y)
y = onehot.transform(y).toarray() 
ic(y.shape)    # (4898, 7)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.995, shuffle=True, random_state=24)

# x 데이터 전처리(scaler)
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

ic(x_train.shape, x_test.shape)   #  x_train.shape: (4873, 11), x_test.shape: (25, 11)
x_train = x_train.reshape(4873, 11, 1)
x_test = x_test.reshape(25, 11, 1)


# 2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Conv1D, LSTM
# model = Sequential()
# model.add(LSTM(240, activation='relu', input_shape=(11,1), return_sequences=True))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(240, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(124, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(7, activation='softmax'))


# 3. 컴파일(EalryStopping), 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_5_MCP.hdf5')

# import time
# start = time.time()
# model.fit(x_train, y_train, epochs=10000, batch_size=512, validation_split=0.0024, callbacks=[es, cp])
# end = time.time() - start

# model.save('./_save/ModelCheckPoint/keras48_5_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_5_model_save.h5')   # save model
model = load_model('./_save/ModelCheckPoint/keras48_5_MCP.hdf5')        # checkpoint

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
'''