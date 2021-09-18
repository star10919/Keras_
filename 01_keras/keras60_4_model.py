from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from icecream import ic

### 증폭하지 않은 fashion_mnist 랑 acc랑 val_acc 비교(=> 과적합을 해결했는지 보려고)


# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

## ImageDataGenerator로 데이터 증폭시키기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest',
    )

#1. ImageDataGenerator를 정의
#2. 파일에서 땡겨오려면 -> flow_from_directory()  :  xy 가 튜플형태로 묶여서 나옴
#3. 데이터에서 땡겨오려면 -> flow()  :  x와 y가 분류되어 있어야 한다.

augment_size=40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)       # x_train[0]에서 아크먼트 사이즈 만큼 랜덤하게 들어감
print(x_train.shape[0])     # 60000
print(randidx)              # [44596 49164  1092 ... 51768  3501 13118]
print(randidx.shape)        # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)       # (40000, 28, 28)

# flow 의 x는 4차원을 받아야 한다!!
x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

                                  # x            # y (y_argumented 넣어도 됨)
x_argumented = train_datagen.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]   # 어차피 x만 추출([0]), y별로 안 중요    # .next() 써주기
print(x_augmented.shape)       # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = one.fit_transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
ic(y_train.shape, y_test.shape)     #  y_train.shape: (100000, 10), y_test.shape: (10000, 10)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, LSTM, Conv1D
model = Sequential()
model.add(Conv2D(128, (2,2), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=512, validation_split=0.001, callbacks=[es])
end = time.time() - start


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])


'''
* onehotencoding
걸린시간 : 256.97893238067627
category : 1.085862636566162
accuracy : 0.8981000185012817

* sparse_categorical_crossentropy
걸린시간 : 81.27210283279419
category : 0.6393763422966003
accuracy : 0.8881000280380249
'''