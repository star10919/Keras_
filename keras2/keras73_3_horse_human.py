# 가장 잘나온 전이학습모델로 이 데이터를 학습시켜서 결과치 도출
# keras59번과의 성능 비교

# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC로 만든 것과 GlobalAveragePooling으로 만든 것 비교

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic


# 1. 데이터

x_train = np.load('./_save/_npy/k59_7_train_x.npy')
y_train = np.load('./_save/_npy/k59_7_train_y.npy')
x_test = np.load('./_save/_npy/k59_7_test_x.npy')
y_test = np.load('./_save/_npy/k59_7_test_y.npy')

# ic(x_train.shape, y_train.shape)    # (2000, 150, 150, 3), y_train.shape: (2000, 2)
# ic(x_test.shape, y_test.shape)      # (661, 150, 150, 3), y_test.shape: (661, 2)


# 2. 모델
transferlearning = Xception(weights='imagenet', include_top=False, input_shape=(150,150,3))   # include_top=False : input_shape 조정 가능

transferlearning.trainable=True
# transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(transferlearning)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.4))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(2, activation='softmax'))


# model.trainable=False   # False: 전체 모델 훈련을 동결한다.(True가 default)

model.summary()

print(len(model.weights))               # 26 -> 30(layer 2개 추가 : 2(w+b)=4)
print(len(model.trainable_weights))     # 0 -> 4



# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=32, validation_data=(x_test, y_test), validation_steps=4, callbacks=[es])
end = time.time() - start


# 4. 평가, 예측
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
hist = model.fit(x_train, y_train, epochs=100, callbacks=[es], steps_per_epoch=32, validation_data=(x_test, y_test), validation_steps=4)


results = model.evaluate(x_test, y_test)
print('category :', results[0])
print('acc :', results[1])



#결과출력
'''
category : 0.7749390602111816
acc : 0.8829268217086792
'''