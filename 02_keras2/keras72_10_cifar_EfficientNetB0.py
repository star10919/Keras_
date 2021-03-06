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
transferlearning = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

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
걸린시간 : 89.90151715278625
category : 0.9830966591835022
accuracy : 0.804099977016449

*trainable = True, GAP
걸린시간 : 82.36472034454346
category : 1.085633635520935
accuracy : 0.7904000282287598

*trainable = False, Flatten
걸린시간 : 90.21846652030945
category : 1.1501272916793823
accuracy : 0.5946999788284302

*trainable = False, Gap
걸린시간 : 102.38142967224121
category : 1.166086196899414
accuracy : 0.5867000222206116



<cifar 100>
*trainable = True, Flatten
걸린시간 : 90.16727018356323
category : 2.3801538944244385
accuracy : 0.5418000221252441

*trainable = True, GAP          ***
걸린시간 : 97.87405109405518
category : 2.4787075519561768
accuracy : 0.5422000288963318

*trainable = False, Flatten
걸린시간 : 149.92054200172424
category : 2.575953483581543
accuracy : 0.35179999470710754

*trainable = False, Gap
걸린시간 : 126.86211800575256
category : 2.588029146194458
accuracy : 0.34529998898506165
'''