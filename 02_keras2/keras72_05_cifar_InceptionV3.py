# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC로 만든 것과 GlobalAveragePooling으로 만든 것 비교

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, UpSampling2D, Dropout
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


# 동결하고, 안하고 비교
# FC를 모델로 하고, GlobalAveragepooling2D으로 하고

# 1. 데이터
# (x_train,y_train), (x_test, y_test) = cifar10.load_data()
# ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
# ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)

(x_train,y_train), (x_test, y_test) = cifar100.load_data()
# ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
# ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)


# ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] - 10개
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# 1-2. y 데이터전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
# ic(y_train.shape, y_test.shape)   # (50000, 10), (10000, 10)


# x 데이터전처리
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)


# 2. 모델
transferlearning = InceptionV3(weights='imagenet', include_top=False, input_shape=(96,96,3))   # include_top=False : input_shape 조정 가능

# transferlearning.trainable=True
transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(UpSampling2D((3,3), input_shape=(32,32,3)))
model.add(Dropout(0.9))
model.add(transferlearning)
model.add(Dropout(0.9))
model.add(Dropout(0.9))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.9))
# model.add(Dense(10))        # *layer 1 추가
# model.add(Dense(10, activation='softmax'))         # *layer 2 추가
model.add(Dense(100, activation='softmax'))


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
model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split=0.012, callbacks=[es])
end = time.time() - start


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])


#결과출력
'''
<cifar 10>
*trainable = True, Flatten
걸린시간 : 461.880916595459
category : 2.3026676177978516
accuracy : 0.10000000149011612

*trainable = True, GAP          ***
걸린시간 : 465.6171078681946
category : 133.4356231689453
accuracy : 0.1136000007390976

*trainable = False, Flatten
걸린시간 : 258.7369191646576
category : 2.4688994884490967
accuracy : 0.11150000244379044

*trainable = False, Gap
걸린시간 : 281.8736536502838
category : 2.4602997303009033
accuracy : 0.10350000113248825



<cifar 100>
*trainable = True, Flatten          ***
걸린시간 : 839.7203347682953
category : 7.593351364135742
accuracy : 0.010200000368058681

*trainable = True, GAP
걸린시간 : 528.8599574565887
category : 4.605205059051514
accuracy : 0.009999999776482582

*trainable = False, Flatten
걸린시간 : 140.64321279525757
category : 5.6840057373046875
accuracy : 0.008299999870359898

*trainable = False, Gap
걸린시간 : 143.16111826896667
category : 5.485367774963379
accuracy : 0.00989999994635582
'''