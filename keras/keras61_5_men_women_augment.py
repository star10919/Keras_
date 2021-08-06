# 훈련데이터를 기존데이터 +20% 더 할 것
# 성과비교
# save_dir도 temp에 넣을 것
# 증폭데이터는 temp에 저장 후 훈련끝난 후 결과 본 뒤 삭제

### 랜덤 ->  [4차원 -> flow] -> concatenate

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
# test_datagen = ImageDataGenerator(rescale=1./255)



# trainGen = imageGen.flow_from_directory(
#     '../data/men_women',
#     target_size=(150, 150),
#     batch_size=2650,
#     class_mode='binary',
#     subset='training',
#     shuffle=False
# )
# # Found 2649 images belonging to 2 classes.
# ic(trainGen[0][0].shape)     # (2649, 150, 150, 3)
# ic(trainGen[0][1].shape)     # (2649,)


# testGen = imageGen.flow_from_directory(
#     '../data/men_women',
#     target_size=(150, 150),
#     batch_size=1000,
#     class_mode='binary',
#     subset='validation',
#     shuffle=False
# )
# #Found 661 images belonging to 2 classes.
# ic(testGen[0][0].shape)     # (661, 150, 150, 3)
# ic(testGen[0][1].shape)     # (661,)


# # 넘파이로 저장
# np.save('./_save/_npy/k59_5_train_x.npy', arr=trainGen[0][0])
# np.save('./_save/_npy/k59_5_train_y.npy', arr=trainGen[0][1])
# np.save('./_save/_npy/k59_5_test_x.npy', arr=testGen[0][0])
# np.save('./_save/_npy/k59_5_test_y.npy', arr=testGen[0][1])


# #============================================================================

x_train = np.load('./_save/_npy/k59_5_train_x.npy')
y_train = np.load('./_save/_npy/k59_5_train_y.npy')
x_test = np.load('./_save/_npy/k59_5_test_x.npy')
y_test = np.load('./_save/_npy/k59_5_test_y.npy')

# ic(x_train.shape, y_train.shape)    # (2649, 150, 150, 3), y_train.shape: (2649,)
# ic(x_test.shape, y_test.shape)      # (662, 150, 150, 3), y_test.shape: (662,)

x_train = x_train[:-1,:,:,:]
x_pred = x_train[-1,:,:,:].reshape(1,150,150,3)
y_train = y_train[:-1]
y_pred = y_train[-1].reshape(1)
ic(x_train.shape, y_train.shape)      # x_train.shape: (2648, 150, 150, 3), y_train.shape: (2648,)
ic(x_pred.shape, y_pred)        # x_pred.shape: (1, 150, 150, 3),  y_pred: 1.0


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation= 'relu'))
model.add(Conv2D(filters=64, kernel_size=(2,2), activation= 'relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))


# 3. 컴파일(ES), 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=32, validation_data=(x_test, y_test), validation_steps=4, callbacks=[es])


results = model.evaluate(x_test, y_test)
print('binary :', results[0])
print('acc :', results[1])

y_predict = model.predict(x_pred)

res = results[1] * 100
print('여자일 확률 :', res, '%')
ic(y_predict, y_predict.shape)

'''
binary : 2.560406446456909
acc : 0.5839636921882629
여자일 확률 : 58.396369218826294 %
'''