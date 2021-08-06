# 훈련데이터를 기존데이터 +20% 더 할 것
# 성과비교
# save_dir도 temp에 넣을 것
# 증폭데이터는 temp에 저장 후 훈련끝난 후 결과 본 뒤 삭제

### 랜덤 ->  [4차원 -> flow] -> concatenate

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../data/brain/train',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../data/brain/test',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)
# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0] # (160, 150, 150, 3)
y_train = xy_train[0][1] # (160, )

x_test = xy_test[0][0] # (120, 150, 150, 3)
y_test = xy_test[0][1] # (120,)

augment_size = 160

randidx = np.random.randint(x_train.shape[0], size=augment_size) # take 40000 feature from train in random
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
ic(x_augmented.shape, y_augmented.shape)    #(160, 150, 150, 3), (160,)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3) # (32, 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3) # (160, 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3) # (120, 150, 150, 3)

x_augmented = train_datagen.flow(x_augmented, 
                                np.zeros(augment_size),
                                batch_size=augment_size,
                                shuffle=False).next()[0]
ic(type(x_augmented), x_augmented.shape)    #<class 'numpy.ndarray'>, (160, 150, 150, 3)

x_train = np.concatenate((x_train, x_augmented)) # (320, 150, 150, 3) 
y_train = np.concatenate((y_train, y_augmented)) # (320,)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(16, (2,2), padding='same', activation='relu', input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. compile train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, verbose=2,
    validation_split=0.2, callbacks=[es], steps_per_epoch=32,
                validation_steps=4)
end_time = time.time() - start_time

# 4. predict eval -> no need to

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-1])
print('val_acc : ',val_acc[-1])
print('loss : ',loss[-1])
print('val_loss : ',val_loss[-1])                             

'''
acc :  0.50390625
val_acc :  0.53125
loss :  0.5
val_loss :  0.6926944255828857
'''