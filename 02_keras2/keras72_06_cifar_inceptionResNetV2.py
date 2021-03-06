# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC로 만든 것과 GlobalAveragePooling으로 만든 것 비교

#결과출력
# cifar 10
# trainable = True, FC : loss=?, acc=?
# trainable = True, GAP : loss=?, acc=?
# trainable = False, FC : loss=?, acc=?
# trainable = Flase, GAP : loss=?, acc=?

# cifar 100
# trainable = True, FC : loss=?, acc=?
# trainable = True, GAP : loss=?, acc=?
# trainable = False, FC : loss=?, acc=?
# trainable = Flase, GAP : loss=?, acc=?

# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False
# FC로 만든 것과 GlobalAveragePooling으로 만든 것 비교

#결과출력
# cifar 10
# trainable = True, FC : loss=?, acc=?
# trainable = True, GAP : loss=?, acc=?
# trainable = False, FC : loss=?, acc=?
# trainable = Flase, GAP : loss=?, acc=?

# cifar 100
# trainable = True, FC : loss=?, acc=?
# trainable = True, GAP : loss=?, acc=?
# trainable = False, FC : loss=?, acc=?
# trainable = Flase, GAP : loss=?, acc=?

# 실습
# cifar10 과  cifar100 으로 모델 만들 것
# trainable=True, False(동결하고, 안하고 비교)
# FC(Flatten)로 만든 것과 GlobalAveragePooling으로 만든 것 비교


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
transferlearning = InceptionResNetV2(weights='imagenet', include_top=False)#, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

# transferlearning.trainable=True
transferlearning.trainable=False    # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(UpSampling2D((3,3), input_shape=(32,32,3)))
model.add(Dropout(0.9))
model.add(transferlearning)
# model.add(Flatten())
model.add(Dropout(0.9))
model.add(GlobalAveragePooling2D())
# model.add(Dense(100))        # *layer 1 추가
model.add(Dropout(0.9))
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
걸린시간 : 1980.6585354804993
category : 2.31429386138916
accuracy : 0.10270000249147415

*trainable = True, GAP          ***
걸린시간 : 594.5940029621124
category : 4.50607442855835
accuracy : 0.11559999734163284

*trainable = False, Flatten
걸린시간 : 318.1835608482361
category : 61.91223907470703
accuracy : 0.08669999986886978

*trainable = False, Gap
걸린시간 : 194.71557927131653
category : 80.87918090820312
accuracy : 0.10000000149011612



<cifar 100>
*trainable = True, Flatten
걸린시간 : 1186.1097021102905
category : 4.605396270751953
accuracy : 0.009999999776482582

*trainable = True, GAP
걸린시간 : 976.053827047348
category : 2177.185302734375
accuracy : 0.009999999776482582

*trainable = False, Flatten         ***
걸린시간 : 315.6615722179413
category : 227.47975158691406
accuracy : 0.013899999670684338

*trainable = False, Gap
걸린시간 : 194.94028329849243
category : 223.3076934814453
accuracy : 0.013700000010430813
'''