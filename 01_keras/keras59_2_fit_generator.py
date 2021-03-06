import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


### fit_generator(fit 할 때, xy가 쌍으로 되어 있을 때 사용)

# 1. 데이터
## ImageDataGenerator로 데이터 증폭시키기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    )

# ImageDataGenerator로 데이터 증폭시키기 - test는 증폭시키지 않음
test_datagen = ImageDataGenerator(rescale=1./255)





## 이미지 x, y로 불러오기
xy_train = train_datagen.flow_from_directory(# x와 y가 동시에 생성됨
    '../data/brain/train',      # 이미지가 있는 폴더 지정이 아닌 상위 폴더(동일한 급의 라벨이 모여있는 폴더까지)로 지정   # train(ad/normal)
    target_size=(150, 150),     # 임의대로 크기 조정
    batch_size=5,               # y 하나의 개수
    class_mode='binary',        # 이상이 있다-라벨 / 이상이 없다-라벨 : 이진분류
    shuffle=False               # 셔플 디폴트 : 트루
)
# Found 160 images belonging to 2 classes.


xy_test = test_datagen.flow_from_directory(
    '../data/brain/test',      
    target_size=(150, 150),     
    batch_size=5,       
    class_mode='binary'         
)
# Found 120 images belonging to 2 classes.




# ## 프린트
# print(xy_train)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001CE349F8550>

# print(xy_train[0])          # x값, y값
# print(xy_train[0][0])       # x값
# print(xy_train[0][1])       # y값
# # print(xy_train[0][2])     # 없음                       
# print(xy_train[0][0].shape, xy_train[0][1].shape)       # (5, 150, 150, 3) (5,)     # 5:배치사이즈

# print(xy_train[31][1])      # 마지막 배치 y
# # print(xy_train[32][1])    # 없음

# print(type(xy_train))           # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))        # <class 'tuple'>
# print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일(ES), 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32,   # 160/5(batch size) = 32
                           validation_data=xy_test,
                           validation_steps=4)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에걸로 시각화 할 것


print('acc :', acc[-1])
print('val_acc :', val_acc[:-1])


'''
acc : 0.612500011920929
'''