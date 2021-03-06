# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC로 만든 것과 GlobalAveragePooling으로 만든 것 비교

# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC(Flatten)로 만든 것과 GlobalAveragePooling으로 만든 것 비교


from numpy.core.fromnumeric import size
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, UpSampling2D, MaxPooling2D, Dropout
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

# 1-2. 데이터전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
# ic(y_train.shape, y_test.shape)   # (50000, 10), (10000, 10)



# 2. 모델
transferlearning = Xception(weights='imagenet', include_top=False, input_shape=(96,96,3))   # include_top=False : input_shape 조정 가능
# *** ValueError: Input size must be at least 71x71; got `input_shape=(32, 32, 3)`   최소 71이어야 하므로 (이왕이면 배수로) 높여주기

# transferlearning.trainable=True
transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(UpSampling2D(size=(3,3), input_shape=(32,32,3)))
model.add(transferlearning)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.9))
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
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=2)

import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split=0.012, callbacks=[es])
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
걸린시간 : 784.2517812252045
category : 0.5313512086868286
accuracy : 0.8776999711990356

*trainable = True, GAP          ***
걸린시간 : 1350.9225783348083
category : 0.4360618591308594
accuracy : 0.9009000062942505

*trainable = False, Flatten
걸린시간 : 355.8347120285034
category : 1.7421749830245972
accuracy : 0.3903999924659729

*trainable = False, Gap
걸린시간 : 260.56324195861816
category : 1.8899943828582764
accuracy : 0.34610000252723694



<cifar 100>
*trainable = True, Flatten          ***
걸린시간 : 975.6288940906525
category : 1.6256664991378784
accuracy : 0.6686000227928162

*trainable = True, GAP
걸린시간 : 1735.1499490737915
category : 2.6373519897460938
accuracy : 0.5401999950408936

*trainable = False, Flatten
걸린시간 : 358.8423228263855
category : 3.917011022567749
accuracy : 0.1031000018119812

*trainable = False, Gap
걸린시간 : 312.2257218360901
category : 4.027853488922119
accuracy : 0.10540000349283218
'''