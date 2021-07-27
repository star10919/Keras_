import numpy as np
from icecream import ic
from sklearn.metrics import r2_score

### concatenate(메소드), Concatenate(클래스) - 앙상블((함수형)모델 합치기)

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1,101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.array(range(1001, 1101))

# ic(x1.shape, x2.shape, y.shape)    # x1.shape: (100, 3),    x2.shape: (100, 3),    y1.shape: (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True, random_state=66)

# ic(x1_train.shape, x1_test.shape, y_train.shape)   # x1_train.shape: (70, 3), x1_test.shape: (30, 3), y_train.shape: (70,)

#2. 모델구성
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3)  #히든레이어

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)  #히든레이어

from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2])   #concatenate(메소드 사용)  #연결만 한거임  
merge1 = Concatenate()([output1, output2])   #Concatenate(클래스 사용)  # 2개니까 리스트 사용
merge2 = Dense(10, name='merge2')(merge1)
merge3 = Dense(5,activation='relu', name='merge3')(merge2)
last_output = Dense(1)(merge3)   #최종 output

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)   # restore_best_weights=True 추가하면 MCP, save model 동일하게 저장됨

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, verbose=1, filepath='./_save/ModelCheckPoint/keras49_MCP.h5')

model.fit([x1_train, x2_train], y_train, epochs=100, validation_split=0.2, batch_size=2, verbose=1, callbacks=[es, mcp])

model.save('./_save/ModelCheckPoint/keras49_model_save.h5')

print('====================== 1. 기본 출력 ======================')

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test)
print('loss :', results[0])   #mse
# print('mae :', results[1])   #metrics['mae]

# R2 결정 계수(평가지표 중 하나) : 정확도와 유사한 지표
y_predict = model.predict([x1_test, x2_test])


r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
print('R2 스코어 : ', r2)

print('====================== 2. load_model ======================')
model2 = load_model('./_save/ModelCheckPoint/keras49_model_save.h5')

results = model2.evaluate([x1_test, x2_test], y_test)
print('loss :', results[0])   #mse
# print('mae :', results[1])   #metrics['mae]

# R2 결정 계수(평가지표 중 하나) : 정확도와 유사한 지표
y_predict = model2.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
print('R2 스코어 : ', r2)

print('=================== 3. model_checkpoint ===================')
model3 = load_model('./_save/ModelCheckPoint/keras49_MCP.h5')

results = model3.evaluate([x1_test, x2_test], y_test)
print('loss :', results[0])   #mse
# print('mae :', results[1])   #metrics['mae]

# R2 결정 계수(평가지표 중 하나) : 정확도와 유사한 지표
y_predict = model3.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
print('R2 스코어 : ', r2)

################ restore_best_weights=False ################
# ====================== 1. 기본 출력 ======================
# loss : 1.363897681236267
# R2 스코어 :  0.9984401583770943
# ====================== 2. load_model ======================
# loss : 1.363897681236267
# R2 스코어 :  0.9984401583770943
# ====================== 2. load_model ======================
# loss : 0.7116487622261047
# R2 스코어 :  0.9991861125415016


################ restore_best_weights=True ###############
# =================== 3. model_checkpoint ===================
# loss : 1085255.25
# R2 스코어 :  -1240.1680641964927
# ====================== 2. load_model ======================
# loss : 1085255.25
# R2 스코어 :  -1240.1680641964927
# ====================== 2. load_model ======================
# loss : 1085255.25
# R2 스코어 :  -1240.1680641964927