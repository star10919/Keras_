from tensorflow.keras.datasets import cifar100, mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.python.distribute.cross_device_ops import HierarchicalCopyAllReduce



# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)-1차원
print(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)
print(np.max(x_train), np.max(x_test))  # 255

# 1-2. x 데이터 전처리 - scaler:2차원에서만 가능
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.    #(50000, 3072)
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.       #(10000, 3072)

from tensorflow.keras.utils import to_categorical   #0,1,2 값이 없어도 무조건 생성/shape유연
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# strategy = tf.distribute.MirroredStrategy()     # strategy 정의 : gpu 2개를 동시에 분산해서 사용 - 분산처리할 때는 배치사이즈 클수록 좋음
# strategy = tf.distribute.MirroredStrategy(cross_device_ops= \
    # tf.distribute.HierarchicalCopyAllReduce()   # 375
    # tf.distribute.ReductionToOneDevice()   # 위에꺼랑 성능차이 거의 없음(아무거나 쓰기)
    # )
# strategy = tf.distribute.MirroredStrategy(
    # devices=['/gpu:0']       # gpu 선택해서 쓸 수 있음
    # devices=['/gpu:1']
    # devices=['/cpu', '/gpu:0']      # cpu, gpu 같이 쓸 수 있음
# )

# strategy = tf.distribute.experimental.CentralStorageStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.RING
    # tf.distribute.experimental.CollectiveCommunication.NCCL
    # tf.distribute.experimental.CollectiveCommunication.AUTO
)



with strategy.scope():

    # 2. 모델구성
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(28, 28, 1))) 
    model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
    model.add(MaxPool2D())        

    model.add(Flatten())                                              
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))


    # 3. 컴파일(ES), 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.25)
end_time = time.time() - start_time


# 4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=128)
print("=============================================")
print("걸린 시간 :", end_time)
print('loss :', results[0])
print('acc :', results[1])

# #plt 시각화
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))   # figure : 판 깔겠다.

# # 1
# plt.subplot(2,1,1)   # subplot : 그림 2개 그리겠다.
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# # 2
# plt.subplot(2,1,2)
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()



'''
걸린 시간 : 50.16567611694336
category : 2.7187631130218506
accuracy : 0.4185999929904938
'''