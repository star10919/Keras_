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
x_train = np.load('./_save/_npy/k59_5_train_x.npy')
y_train = np.load('./_save/_npy/k59_5_train_y.npy')
x_test = np.load('./_save/_npy/k59_5_test_x.npy')
y_test = np.load('./_save/_npy/k59_5_test_y.npy')

# ic(x_train.shape, y_train.shape)    # (2649, 150, 150, 3), y_train.shape: (2649,)
# ic(x_test.shape, y_test.shape)      # (662, 150, 150, 3), y_test.shape: (662,)

x_train = x_train[:-1,:,:,:]
x_pred = x_train[-1,:,:,:].reshape(1,150,150,3)
y_train = y_train[:-1]
y_pred = y_train[-1].reshape(1)
ic(x_train.shape, y_train.shape)      # x_train.shape: (2648, 150, 150, 3), y_train.shape: (2648,)
ic(x_pred.shape, y_pred)        # x_pred.shape: (1, 150, 150, 3),  y_pred: 1.0



# 2. 모델
transferlearning = ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

# transferlearning.trainable=True
transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(transferlearning)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(100))        # *layer 1 추가
# model.add(Dense(10, activation='softmax'))         # *layer 2 추가
model.add(Dense(1, activation='sigmoid'))


# model.trainable=False   # False: 전체 모델 훈련을 동결한다.(True가 default)

model.summary()

print(len(model.weights))               # 26 -> 30(layer 2개 추가 : 2(w+b)=4)
print(len(model.trainable_weights))     # 0 -> 4



# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1024, validation_split=0.012, callbacks=[es])
end = time.time() - start


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])


#결과출력
'''

'''