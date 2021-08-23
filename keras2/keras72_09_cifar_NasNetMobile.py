# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC로 만든 것과 GlobalAveragePooling으로 만든 것 비교


from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, UpSampling2D
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
transferlearning = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224,224,3))   # include_top=False : input_shape 조정 가능
# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).

transferlearning.trainable=True
# transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(UpSampling2D((7,7), input_shape=(32,32,3)))
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
es = EarlyStopping(monitor='val_loss', mode='min', patience=2, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=13, batch_size=50, validation_split=0.012, callbacks=[es], verbose=2)
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
걸린시간 : 2501.4734745025635
category : 7.954424858093262
accuracy : 0.09989999979734421

*trainable = True, GAP
걸린시간 : 5863.720641136169
category : 2.4339160919189453
accuracy : 0.45399999618530273

*trainable = False, Flatten
걸린시간 : 303.26136565208435
category : 1.8013070821762085
accuracy : 0.38179999589920044

*trainable = False, Gap
걸린시간 : 417.87746381759644
category : 1.7630566358566284
accuracy : 0.3734000027179718



<cifar 100>
*trainable = True, Flatten
걸린시간 : 1596.9447824954987
category : 8.750218391418457
accuracy : 0.01269999984651804

*trainable = True, GAP
걸린시간 : 2823.2439806461334
category : 6.423914432525635
accuracy : 0.03869999945163727

*trainable = False, Flatten
걸린시간 : 305.3029828071594
category : 4.757012367248535
accuracy : 0.13249999284744263

*trainable = False, Gap
걸린시간 : 477.19129395484924
category : 3.78521990776062
accuracy : 0.1429000049829483
'''