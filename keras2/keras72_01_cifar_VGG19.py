# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC(Flatten)로 만든 것과 GlobalAveragePooling으로 만든 것 비교


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
transferlearning = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

# transferlearning.trainable=True
transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(transferlearning)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))        # *layer 1 추가
# model.add(Dense(10, activation='softmax'))
model.add(Dense(100, activation='softmax'))         # *layer 2 추가

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
*trainable = True, Flatten       ***
걸린시간 : 375.82809257507324
category : 0.8650935292243958
accuracy : 0.7573999762535095

*trainable = True, GAP
걸린시간 : 347.42102122306824
category : 1.0038697719573975
accuracy : 0.7361000180244446

*trainable = False, Flatten
걸린시간 : 51.21760630607605
category : 1.4738796949386597
accuracy : 0.5503000020980835

*trainable = False, Gap
걸린시간 : 57.98673677444458
category : 1.467638611793518
accuracy : 0.5493000149726868



<cifar 100>
*trainable = True, Flatten
걸린시간 : 404.26293540000916
category : 2.779270648956299
accuracy : 0.3151000142097473

*trainable = True, GAP
걸린시간 : 71.34248614311218
category : 4.605197429656982
accuracy : 0.009999999776482582

*trainable = False, Flatten
걸린시간 : 97.46437883377075
category : 2.939389944076538
accuracy : 0.32420000433921814

*trainable = False, Gap
걸린시간 : 91.73789048194885
category : 2.94810152053833
accuracy : 0.32829999923706055
'''