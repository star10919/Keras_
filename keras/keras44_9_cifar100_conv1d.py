#overfit을 극복하자
'''
1. 전체 훈련 데이터가 많이 있을수록 좋다
2. 정규화(normarlization) 고
-> 전처리 할 때 scailing에서 했던 정규화로는 부족
-> layer에서 layer로 넘어갈 때 활성화 함수로 감쌀때 정규화 시켜줘야 한다는 의미
-> layer별로 정규화를 시켜주는것은 어떠냐는 의미 
3. dropout
: 각 layer마다 몇개의 node를 빼고 계산했을 때 과적합이 줄어듦었음을 알 수 있다
: 이 때 머신은 랜덤하게 제외시킬 노트를 일정비율로 솎아준다.
'''
import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.python.keras.layers.recurrent import LSTM

### GlobalAveragePooling2D

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
print(np.unique(y_train)) 
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
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
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
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_split=0.2, shuffle=True, batch_size=512)
end_time = time.time() - start_time


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("걸린시간 :", end_time)
print('category :', loss[0])
print('accuracy :', loss[1])


# 시각화 
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1) # 2개의 플롯을 할건데, 1행 1열을 사용하겠다는 의미 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2) # 2개의 플롯을 할건데, 1행 2열을 사용하겠다는 의미 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()


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
'''