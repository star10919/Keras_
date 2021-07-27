######### 문제설명 ###########
# 시 고 저 종 거래량만으로 컬럼 구성
# 행은 2011년 1월 3일 이후 데이터로 구성

# 삼성과 sk 각각 5개의 컬럼씩
# 앙상블하고
# 1. 금요일 종가 맞추기 (금 오전 9시 전까지)
# 2. 월요일 시가 맞추기 (월 오전 9시 전까지)

# 제출파일 : 1. 원소스      2. 가중치를 불러올 수 있는 소스       3. 가중치
# 파일명 : 본인이름 이니셜 + 삼성전자 종가.py   # LHI1_57000.py
# 메일제목 : 본인이름 주가                     # 이해인 1차 57000원
############################################################################


# 1. 데이터
import pandas as pd
from icecream import ic
import numpy as np
from sklearn.metrics import r2_score
import time

raw_samsung = pd.read_csv('./samsung/_data_save/samsung_stock.csv', header=0, encoding='CP949')
raw_sk = pd.read_csv('./samsung/_data_save/sk_stock.csv', header=0, encoding='CP949')
# ic(data_samsung)
# ic(data_sk)

df_samsung = pd.DataFrame(raw_samsung)   # 데이터프레임으로 전환
df_sk = pd.DataFrame(raw_sk)
# ic(df_samsung, df_sk)

samsung = df_samsung[['시가','고가','저가','거래량','종가']]   # 열 추출
sk = df_sk[['시가','고가','저가','거래량','종가']]

samsung = samsung.iloc[0:2601]     # 행 추출
sk = sk.iloc[0:2601]
# ic(samsung.shape, sk.shape)   # samsung.shape: (2601, 5), sk.shape: (2601, 5)

samsung = samsung.sort_index(ascending=False)   # 오름차순 정렬
sk = sk.sort_index(ascending=False)
ic(samsung, sk)     #(2601, 5), (2601, 5)

'''
ic| samsung:            시가       고가       저가         거래량       종가
             2600  19100.0  19320.0  19000.0  13278100.0  19160.0
             2599  19120.0  19220.0  18980.0  13724400.0  19160.0
             2598  19100.0  19100.0  18840.0  16811200.0  18840.0
             2597  18840.0  18980.0  18460.0  19374400.0  18600.0
             2596  18300.0  18580.0  18280.0  23172350.0  18420.0
             ...       ...      ...      ...         ...      ...
             4     79800.0  80600.0  79500.0  13766279.0  80600.0
             3     80100.0  80100.0  79500.0  10859399.0  79800.0
             2     79100.0  79200.0  78800.0  13155414.0  79000.0
             1     78500.0  79000.0  78400.0  12456646.0  79000.0
             0     79000.0  79100.0  78500.0  12355296.0  78500.0

    sk:             시가        고가        저가         거래량        종가
        2600   25000.0   25300.0   24800.0  15107759.0   25300.0
        2599   25400.0   25600.0   25050.0  10931161.0   25600.0
        2598   25450.0   26300.0   25200.0  14139328.0   25900.0
        2597   26250.0   26600.0   25800.0  14287171.0   26100.0
        2596   26100.0   26450.0   25700.0   7024336.0   26100.0
        ...        ...       ...       ...         ...       ...
        4     123500.0  124000.0  122500.0   1500981.0  123500.0
        3     122000.0  122500.0  120500.0   2905546.0  121500.0
        2     119000.0  120000.0  118500.0   2066638.0  119000.0
        1     117500.0  119500.0  117500.0   2070074.0  118500.0
        0     119500.0  120000.0  116500.0   2864601.0  117000.0
'''
samsung = samsung.to_numpy()   # 넘파이로 변환
sk = sk.to_numpy()

size = 5

def split_xy(dataset, size):
    aaa = []      # 3차원
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size), :]     # 2차원
        aaa.append(subset)
    return np.array(aaa)


split_samsung = split_xy(samsung, size)
split_sk = split_xy(sk, size)

ic(split_samsung.shape, split_sk.shape)      # split_samsung.shape: (2597, 5, 5), split_sk.shape: (2597, 5, 5)

x_samsung = split_samsung[:-1,:,:]
x_sk = split_sk[:-1,:,:]
ic(x_samsung.shape, x_sk.shape)              # x_samsung.shape: (2596, 5, 5),  x_sk.shape: (2596, 5, 5)


x_samsung_pred = split_samsung[-1, :]
x_sk_pred = split_sk[-1, :]

ic(x_samsung_pred.shape)    # x_samsung_pred.shape: (5, 5)
print('******************************')
y_samsung = samsung[5:, 0]    # 삼성시가
ic(y_samsung)     #  y_samsung.shape: (2596,)


x_samsung = x_samsung.reshape(2596,25)
x_sk = x_sk.reshape(2596,25)
y_samsung = y_samsung.reshape(-1,1)
x_samsung_pred = x_samsung_pred.reshape(1,25)
x_sk_pred = x_sk_pred.reshape(1,25)

from sklearn.model_selection import train_test_split
sam_x_train, sam_x_test, sam_y_train, sam_y_test = train_test_split(x_samsung, y_samsung, shuffle=False, train_size=0.8)
sk_x_train, sk_x_test, sk_y_train, sk_y_test = train_test_split(x_sk, y_samsung, shuffle=False, train_size=0.8)

# 1-2. x 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer
scaler = StandardScaler()
sam_x_train = scaler.fit_transform(sam_x_train)
sam_x_test = scaler.transform(sam_x_test)
x_samsung_pred = scaler.transform(x_samsung_pred)
scaler2= StandardScaler()
sk_x_train = scaler2.fit_transform(sk_x_train)
sk_x_test = scaler2.transform(sk_x_test)
x_sk_pred = scaler2.transform(x_sk_pred)


sam_x_train = sam_x_train.reshape(sam_x_train.shape[0], 5, 5)   # 3차원으로 reshape
sam_x_test = sam_x_test.reshape(sam_x_test.shape[0], 5, 5)
x_samsung_pred = x_samsung_pred.reshape(x_samsung_pred.shape[0], 5, 5)
x_sk_pred = x_sk_pred.reshape(x_sk_pred.shape[0], 5, 5)
sk_x_train = sk_x_train.reshape(sk_x_train.shape[0], 5, 5)
sk_x_test = sk_x_test.reshape(sk_x_test.shape[0], 5, 5)
print('************************************************************************************')
ic(sam_x_train.shape, sam_x_test.shape, x_samsung_pred.shape, sk_x_train.shape, sk_x_test.shape)
'''
    sam_x_train.shape: (2076, 5, 5)
    sam_x_test.shape: (520, 5, 5)
    x_samsung_pred.shape: (1, 5, 5)
    sk_x_train.shape: (2076, 5, 5)
    sk_x_test.shape: (520, 5, 5)
'''


# 2. 모델 구성
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPool2D, LSTM, Flatten, GlobalAveragePooling2D, Input, Dropout, MaxPool1D

# 모델1 - samsung
input1 = Input(shape=(5, 5))
lstm = LSTM(12, activation='relu', return_sequences=True)(input1)
conv = Conv1D(256, 3, activation='relu')(lstm)
# conv = MaxPool1D()(conv)
# conv = Dropout(0.2)(conv)
flat = Flatten()(conv)
dense = Dense(128, activation='relu')(flat)
# dense = Dense(10, activation='relu')(dense)
dense = Dense(128, activation='relu')(dense)
dense = Dense(128, activation='relu')(dense)
dense = Dense(128, activation='relu')(dense)
output1 = Dense(1)(dense)

# 모델2 - sk
input2 = Input(shape=(5, 5))
lstm = LSTM(12, activation='relu', return_sequences=True)(input2)
conv = Conv1D(150, 3, activation='relu')(lstm)
# conv = MaxPool1D()(conv)
# conv = Dropout(0.2)(conv)
flat = Flatten()(conv)
dense = Dense(128, activation='relu')(flat)
# dense = Dense(10, activation='relu')(dense)
dense = Dense(128, activation='relu')(dense)
dense = Dense(128, activation='relu')(dense)
dense = Dense(128, activation='relu')(dense)
output2 = Dense(1)(dense)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(10, activation='relu')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

# model.summary()



# 3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=8, verbose=1, restore_best_weights=True)


# import datetime
# date = datetime.datetime.now()
# date_time = date.strftime("%m%d_%H%M")

# filepath = './_save/ModelCheckPoint/'
# filename = '.{epoch:04d}_{val_loss:4f}.hdf5'
# modelpath = "".join([filepath, "SSSS_", date_time, "_", filename])

cp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
            save_best_only=True, 
            filepath= './_save/ModelCheckPoint/pred_stock_MCP14.py')

model.fit([sam_x_train, sk_x_train], sam_y_train, epochs=150, batch_size=50, validation_split=0.02, callbacks=[es, cp])


model.save('./_save/pred_stock_model_save14.h5')
model.save_weights('./_save/pred_stock_weight_save14.h5')

# 4. 평가, 예측
results = model.evaluate([sam_x_test, sk_x_test], [sam_y_test, sam_y_test])
print('loss :', results)

y_predict = model.predict([x_samsung_pred, x_sk_pred])
ic(y_predict)


'''
*'./_save/pred_stock_model_save1.h5'
Epoch 00026: val_loss did not improve from 3333148.25000
Epoch 00026: early stopping
17/17 [==============================] - 0s 7ms/step - loss: 8219378.5000
loss : 8219378.5
ic| y_predict: array([[83043.85]], dtype=float32)


*'./_save/pred_stock_model_save2.h5'
Epoch 00041: val_loss did not improve from 2154355.50000
Epoch 00041: early stopping
17/17 [==============================] - 0s 7ms/step - loss: 9386415.0000
loss : 9386415.0
ic| y_predict: array([[82653.84]], dtype=float32)

*'./_save/pred_stock_model_save3.h5'
Epoch 00030: val_loss did not improve from 3047323.50000
Epoch 00030: early stopping
17/17 [==============================] - 0s 7ms/step - loss: 9296757.0000
loss : 9296757.0
ic| y_predict: array([[78760.78]], dtype=float32)

*'./_save/pred_stock_model_save4.h5'
Epoch 00038: val_loss did not improve from 2378376.00000
Epoch 00038: early stopping
17/17 [==============================] - 0s 8ms/step - loss: 8245199.0000
loss : 8245199.0
ic| y_predict: array([[81655.03]], dtype=float32)

*'./_save/pred_stock_model_save5.h5'
Epoch 00047: val_loss did not improve from 2645629.00000
Epoch 00047: early stopping
17/17 [==============================] - 0s 7ms/step - loss: 11286671.0000
loss : 11286671.0
ic| y_predict: array([[82351.12]], dtype=float32)

*'./_save/pred_stock_model_save6.h5'
Epoch 00100: val_loss improved from 470029.71875 to 418080.34375, saving model to ./_save/ModelCheckPoint\pred_stock_MCP6.py
17/17 [==============================] - 0s 8ms/step - loss: 1959667.0000
loss : 1959667.0
ic| y_predict: array([[78909.66]], dtype=float32)

*'./_save/pred_stock_model_save7.h5'
Epoch 00031: val_loss did not improve from 886492.56250
Epoch 00031: early stopping
17/17 [==============================] - 0s 8ms/step - loss: 11951948.0000
loss : 11951948.0
ic| y_predict: array([[78941.61]], dtype=float32)

*'./_save/pred_stock_model_save8.h5
Epoch 00071: val_loss did not improve from 688020.75000
Epoch 00071: early stopping
17/17 [==============================] - 0s 8ms/step - loss: 3166693.2500
loss : 3166693.25
ic| y_predict: array([[79501.78]], dtype=float32)

*'./_save/pred_stock_model_save9.h5'
Epoch 00086: val_loss did not improve from 206580.32812
Epoch 00086: early stopping
17/17 [==============================] - 0s 8ms/step - loss: 3041688.2500
loss : 3041688.25
ic| y_predict: array([[80449.23]], dtype=float32)

*'./_save/pred_stock_model_save10.h5'
Epoch 00150: val_loss did not improve from 176996.26562
17/17 [==============================] - 0s 8ms/step - loss: 2163886.2500
loss : 2163886.25
ic| y_predict: array([[79906.57]], dtype=float32)

*'./_save/pred_stock_model_save11.h5'
Epoch 00150: val_loss improved from 201507.67188 to 196258.00000, saving model to ./_save/ModelCheckPoint\pred_stock_MCP11.py
17/17 [==============================] - 0s 8ms/step - loss: 3713771.2500
loss : 3713771.25
ic| y_predict: array([[79416.09]], dtype=float32)

*'./_save/pred_stock_model_save12.h5'
Epoch 00126: val_loss did not improve from 296269.00000
Epoch 00126: early stopping
17/17 [==============================] - 0s 8ms/step - loss: 2424225.7500
loss : 2424225.75
ic| y_predict: array([[79440.53]], dtype=float32)

*'./_save/pred_stock_model_save13.h5'
Epoch 00060: val_loss did not improve from 1390906.50000
Epoch 00060: early stopping
17/17 [==============================] - 0s 9ms/step - loss: 8180446.0000
loss : 8180446.0
ic| y_predict: array([[80371.01]], dtype=float32)
'''