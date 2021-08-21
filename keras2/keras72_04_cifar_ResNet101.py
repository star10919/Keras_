# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC로 만든 것과 GlobalAveragePooling으로 만든 것 비교

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
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

# 1-2. 데이터전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
# ic(y_train.shape, y_test.shape)   # (50000, 10), (10000, 10)



# 2. 모델
transferlearning = ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

# transferlearning.trainable=True
transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(transferlearning)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))        # *layer 1 추가
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
model.fit(x_train, y_train, epochs=100, batch_size=1024, validation_split=0.012, callbacks=[es])
end = time.time() - start


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])


#결과출력
'''
<cifar 10>
*trainable = True, Flatten          ***
걸린시간 : 205.41930079460144
category : 1.017120599746704
accuracy : 0.7642999887466431

*trainable = True, GAP
걸린시간 : 254.53516626358032
category : 1.020358681678772
accuracy : 0.6646999716758728

*trainable = False, Flatten
걸린시간 : 71.1706657409668
category : 1.3452637195587158
accuracy : 0.5828999876976013

*trainable = False, Gap
걸린시간 : 85.1605474948883
category : 1.2910833358764648
accuracy : 0.5972999930381775



<cifar 100>
*trainable = True, Flatten
걸린시간 : 205.15047550201416
category : 3.574073553085327
accuracy : 0.3864000141620636

*trainable = True, GAP          ***
걸린시간 : 241.72242832183838
category : 2.7042086124420166
accuracy : 0.4772999882698059

*trainable = False, Flatten
걸린시간 : 67.52339744567871
category : 2.83028507232666
accuracy : 0.35569998621940613

*trainable = False, Gap
걸린시간 : 72.30664587020874
category : 2.8789143562316895
accuracy : 0.35089999437332153
'''