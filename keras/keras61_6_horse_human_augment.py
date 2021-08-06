# 훈련데이터를 기존데이터 +20% 더 할 것
# 성과비교
# save_dir도 temp에 넣을 것
# 증폭데이터는 temp에 저장 후 훈련끝난 후 결과 본 뒤 삭제

### 랜덤 ->  [4차원 -> flow] -> concatenate

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic


## ImageDataGenerator로 데이터 증폭시키기
imageGen = ImageDataGenerator(
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
# test_datagen = ImageDataGenerator(rescale=1./255)




# trainGen = imageGen.flow_from_directory(
#     '../data/horse-or-human',
#     target_size=(150, 150),
#     batch_size=2000,
#     class_mode='categorical',
#     subset='training'
# )
# # Found 822 images belonging to 2 classes.
# ic(trainGen[0][0].shape)     #  (822, 150, 150, 3)
# ic(trainGen[0][1].shape)     #  (822, 2)


# testGen = imageGen.flow_from_directory(
#     '../data/horse-or-human',
#     target_size=(150, 150),
#     batch_size=1000,
#     class_mode='categorical',
#     subset='validation'
# )
# # Found 205 images belonging to 2 classes.
# ic(testGen[0][0].shape)     # (205, 150, 150, 3)
# ic(testGen[0][1].shape)     # (205, 2)




# # 넘파이로 저장
# np.save('./_save/_npy/k59_7_train_x.npy', arr=trainGen[0][0])
# np.save('./_save/_npy/k59_7_train_y.npy', arr=trainGen[0][1])
# np.save('./_save/_npy/k59_7_test_x.npy', arr=testGen[0][0])
# np.save('./_save/_npy/k59_7_test_y.npy', arr=testGen[0][1])


#============================================================================

x_train = np.load('./_save/_npy/k59_7_train_x.npy')
y_train = np.load('./_save/_npy/k59_7_train_y.npy')
x_test = np.load('./_save/_npy/k59_7_test_x.npy')
y_test = np.load('./_save/_npy/k59_7_test_y.npy')

ic(x_train.shape, y_train.shape)    # (822, 150, 150, 3), (822, 2)
ic(x_test.shape, y_test.shape)      # (205, 150, 150, 3), (205, 2)



augment_size = 164

randidx = np.random.randint(x_train.shape[0], size=augment_size) # take 40000 feature from train in random
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
ic(x_augmented.shape, y_augmented.shape)    #(164, 150, 150, 3), (530,)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3) # (164, 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3) # (822, 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3) # (205, 150, 150, 3)
print('****************** 2 *****************')
ic(x_augmented.shape, x_train.shape, x_test.shape)

x_augmented = imageGen.flow(x_augmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]
ic(type(x_augmented), x_augmented.shape)    #<class 'numpy.ndarray'>, (164, 150, 150, 3)



x_train = np.concatenate((x_train, x_augmented)) # (986, 150, 150, 3) 
y_train = np.concatenate((y_train, y_augmented)) # (986, 2)
print('****************** 3 *****************')
ic(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
'''
    x_train.shape: (986, 150, 150, 3)
    y_train.shape: (986, 2)
    x_test.shape: (205, 150, 150, 3)
    y_test.shape: (205, 2)
'''


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150,150,3), padding='same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))


# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=32, validation_data=(x_test, y_test), validation_steps=4)


results = model.evaluate(x_test, y_test)
print('category :', results[0])
print('acc :', results[1])


'''
category : 3.395993709564209
acc : 0.712195098400116

* augment 164(+20%)
category : 4.074365139007568
acc : 0.6048780679702759
'''