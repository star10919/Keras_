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
transferlearning = DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

transferlearning.trainable=True
# transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

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
걸린시간 : 156.5967559814453
category : 0.9979033470153809
accuracy : 0.7994999885559082

*trainable = True, GAP
걸린시간 : 125.64718389511108
category : 1.0339107513427734
accuracy : 0.7828999757766724

*trainable = False, Flatten
걸린시간 : 93.53797006607056
category : 1.3185454607009888
accuracy : 0.5407999753952026

*trainable = False, Gap
걸린시간 : 70.61392450332642
category : 1.3340404033660889
accuracy : 0.5432999730110168



<cifar 100>
*trainable = True, Flatten
걸린시간 : 120.80545330047607
category : 2.4021618366241455
accuracy : 0.525600016117096

*trainable = True, GAP          ***
걸린시간 : 136.15213179588318
category : 2.548996925354004
accuracy : 0.5411999821662903

*trainable = False, Flatten
걸린시간 : 76.2135157585144
category : 2.9471278190612793
accuracy : 0.29739999771118164

*trainable = False, Gap
걸린시간 : 86.39826345443726
category : 2.880398750305176
accuracy : 0.3109000027179718
'''