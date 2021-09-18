# 훈련데이터를 기존데이터 +20% 더 할 것
# 성과비교
# save_dir도 temp에 넣을 것
# 증폭데이터는 temp에 저장 후 훈련끝난 후 결과 본 뒤 삭제

### 랜덤 ->  [4차원 -> flow] -> concatenate

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic


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
    validation_split=0.2
    )
test_datagen = ImageDataGenerator(rescale=1./255)




# trainGen = train_datagen.flow_from_directory(
#     '../data/cat_dog/training_set/training_set',
#     target_size=(150, 150),
#     batch_size=2000,
#     class_mode='categorical'
# )
# # Found 8005 images belonging to 2 classes.
# ic(trainGen[0][0].shape)     # (2000, 150, 150, 3)
# ic(trainGen[0][1].shape)     # (2000, 1)


# testGen = test_datagen.flow_from_directory(
#     '../data/cat_dog/test_set/test_set',
#     target_size=(150, 150),
#     batch_size=1000,
#     class_mode='categorical',
# )
# # FFound 2023 images belonging to 2 classes.
# ic(testGen[0][0].shape)     # (1000, 150, 150, 3)
# ic(testGen[0][1].shape)     # (1000, 2)




# # 넘파이로 저장
# np.save('./_save/_npy/k59_8_train_x.npy', arr=trainGen[0][0])
# np.save('./_save/_npy/k59_8_train_y.npy', arr=trainGen[0][1])
# np.save('./_save/_npy/k59_8_test_x.npy', arr=testGen[0][0])
# np.save('./_save/_npy/k59_8_test_y.npy', arr=testGen[0][1])


#============================================================================

x_train = np.load('./_save/_npy/k59_8_train_x.npy')
y_train = np.load('./_save/_npy/k59_8_train_y.npy')
x_test = np.load('./_save/_npy/k59_8_test_x.npy')
y_test = np.load('./_save/_npy/k59_8_test_y.npy')
print('****************** 1 *****************')
ic(x_train.shape, y_train.shape)    # (2000, 150, 150, 3), (2000, 2)
ic(x_test.shape, y_test.shape)      # (1000, 150, 150, 3), (1000, 2)



augment_size = 400

randidx = np.random.randint(x_train.shape[0], size=augment_size) # take 40000 feature from train in random
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
ic(x_augmented.shape, y_augmented.shape)    #(400, 150, 150, 3), (400, 2)

print('****************** 2 *****************')
ic(x_augmented.shape, x_train.shape, x_test.shape)

x_augmented = train_datagen.flow(x_augmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]
ic(type(x_augmented), x_augmented.shape)    #<class 'numpy.ndarray'>, (400, 150, 150, 3)



x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print('****************** 3 *****************')
ic(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
'''
    x_train.shape: (2400, 150, 150, 3)
    y_train.shape: (2400, 2)
    x_test.shape: (1000, 150, 150, 3)
    y_test.shape: (1000, 2)
'''



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(150,150,3), padding='same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=32, validation_data=(x_test, y_test), validation_steps=4)
end = time.time() - start

results = model.evaluate(x_test, y_test)
print("걸린시간 :", end)
print('category :', results[0])
print('acc :', results[1])


'''
걸린시간 : 132.89668941497803
category : 7.166528701782227
acc : 0.531000018119812

*augment 400(+20%)
걸린시간 : 328.7054591178894
category : 8.031929969787598
acc : 0.5070000290870667
'''