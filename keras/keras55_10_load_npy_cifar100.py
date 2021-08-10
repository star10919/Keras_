import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Conv1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 

### 데이터 로드하기

x_train_cifar100 = np.load('./_save/_npy/k55_x_train_cifar100.npy')
x_test_cifar100 = np.load('./_save/_npy/k55_x_test_cifar100.npy')
y_train_cifar100 = np.load('./_save/_npy/k55_y_train_cifar100.npy')
y_test_cifar100 = np.load('./_save/_npy/k55_y_test_cifar100.npy')

ic(x_train_cifar100)
ic(x_test_cifar100)
ic(y_train_cifar100)
ic(y_test_cifar100)
ic(x_train_cifar100.shape, x_test_cifar100.shape, y_train_cifar100.shape, y_test_cifar100.shape)

'''
 x_train_cifar100.shape: (50000, 32, 32, 3)
    x_test_cifar100.shape: (10000, 32, 32, 3)
    y_train_cifar100.shape: (50000, 1)
    y_test_cifar100.shape: (10000, 1)
'''

# np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=x_train_cifar100)
# np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=x_test_cifar100)
# np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=y_train_cifar100)
# np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=y_test_cifar100)

x_train = x_train_cifar100.reshape(50000, 32 * 32 * 3)
x_test = x_test_cifar100.reshape(10000, 32 * 32 * 3)
print(np.unique(y_train_cifar100)) 
# 전처리 하기 -> scailing
# 단, 2차원 데이터만 가능하므로 4차원 -> 2차원
# x_train = x_train/255.
# x_test = x_test/255.
print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

# 1-2. x 데이터 전처리
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(50000, 32 , 96)
x_test = x_test.reshape(10000, 32, 96)

# 1-3. y 데이터 전처리 -> one-hot-encoding
y_train = to_categorical(y_train_cifar100)
y_test = to_categorical(y_test_cifar100)
print(y_train.shape, y_test.shape)


# 2. 모델 구성(GlobalAveragePooling2D 사용)
model = Sequential()
model.add(LSTM(128, input_shape=(32, 96), activation='relu', return_sequences=True))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))    
model.add(Dense(128, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))


# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_9_MCP.hdf5')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es, cp], validation_split=0.2, shuffle=True, batch_size=512)
end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_9_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_9_model_save.h5')           # save model
# model = load_model('./_save/ModelCheckPoint/keras48_9_MCP.hdf5')                # checkpoint

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("걸린시간 :", end_time)
print('category :', loss[0])
print('accuracy :', loss[1])


# # 시각화 
# plt.figure(figsize=(9,5))

# # 1
# plt.subplot(2, 1, 1) # 2개의 플롯을 할건데, 1행 1열을 사용하겠다는 의미 
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# # 2
# plt.subplot(2, 1, 2) # 2개의 플롯을 할건데, 1행 2열을 사용하겠다는 의미 
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()


'''
*cnn + Standard
걸린시간:  402.19969177246094
category:  3.1498517990112305
accuracy:  0.35659998655319214

*patience, batch 줄이고 validation 늘렸을 때 
category:  3.038492441177368
accuracy:  0.3449999988079071

*validation 높이고, modeling 수정
걸린시간:  174.78156971931458
category:  3.2364041805267334
accuracy:  0.37290000915527344

*batch_size 더 줄였을때 128-> 64
걸린시간:  207.46179294586182
category:  3.2013678550720215
accuracy:  0.3716000020503998

*batch_size 64 -> 256 늘렸을 떄
걸린시간:  151.7369945049286
category:  2.806745767593384
accuracy:  0.3878999948501587

*dropout 실행
걸린시간:  660.569277048111
category:  2.0348639488220215
accuracy:  0.4722000062465668

*GlobalAveragePooling
걸린시간:  495.05068159103394
category:  1.959947109222412
accuracy:  0.48559999465942383

*dnn
걸린시간 : 18.44869613647461
category : 3.656463623046875
accuracy : 0.211899995803833

*LSTM
걸린시간 : 438.62733125686646
category : 3.5003504753112793
accuracy : 0.22840000689029694

*LSTM + Conv1D
걸린시간 : 325.8680090904236
category : 3.3531265258789062
accuracy : 0.2623000144958496

*save model
걸린시간 : 328.9038350582123
category : 3.4573137760162354
accuracy : 0.25780001282691956

*checkpoint
category : 3.124910831451416
accuracy : 0.2451999932527542

*load_npy
걸린시간 : 144.05347609519958
category : 3.4703280925750732
accuracy : 0.26649999618530273
'''