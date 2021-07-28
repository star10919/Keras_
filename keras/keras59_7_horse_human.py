# 실습
# 말과 사람 데이터셋으로 완성하시오!

# 이진분류이지만 다중분류(softmax)로 풀 것

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic


# ## ImageDataGenerator로 데이터 증폭시키기
# imageGen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest',
#     validation_split=0.2
#     )
# # test_datagen = ImageDataGenerator(rescale=1./255)




# trainGen = imageGen.flow_from_directory(
#     '../data/rps',
#     target_size=(150, 150),
#     batch_size=2000,
#     class_mode='categorical',
#     subset='training'
# )
# # Found 2016 images belonging to 3 classes.
# ic(trainGen[0][0].shape)     # trainGen[0][0].shape: (2000, 150, 150, 3)
# ic(trainGen[0][1].shape)     # trainGen[0][1].shape: (2000, 3)


# testGen = imageGen.flow_from_directory(
#     '../data/rps',
#     target_size=(150, 150),
#     batch_size=1000,
#     class_mode='categorical',
#     subset='validation'
# )
# # Found 504 images belonging to 3 classes.
# ic(testGen[0][0].shape)     # testGen[0][0].shape: (504, 150, 150, 3)
# ic(testGen[0][1].shape)     # testGen[0][1].shape: (504, 3)




# # 넘파이로 저장
# np.save('./_save/_npy/k59_6_train_x.npy', arr=trainGen[0][0])
# np.save('./_save/_npy/k59_6_train_y.npy', arr=trainGen[0][1])
# np.save('./_save/_npy/k59_6_test_x.npy', arr=testGen[0][0])
# np.save('./_save/_npy/k59_6_test_y.npy', arr=testGen[0][1])


#============================================================================

x_train = np.load('./_save/_npy/k59_6_train_x.npy')
y_train = np.load('./_save/_npy/k59_6_train_y.npy')
x_test = np.load('./_save/_npy/k59_6_test_x.npy')
y_test = np.load('./_save/_npy/k59_6_test_y.npy')

# ic(x_train.shape, y_train.shape)    # (2000, 150, 150, 3), y_train.shape: (2000, 2)
# ic(x_test.shape, y_test.shape)      # (661, 150, 150, 3), y_test.shape: (661, 2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=(150,150,3)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(rate=0.3))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(rate=0.3))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=32, validation_data=(x_test, y_test), validation_steps=4)


results = model.evaluate(x_test, y_test)
print('binary :', results[0])
print('acc :', results[1])