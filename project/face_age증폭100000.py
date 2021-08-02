### 랜덤 ->  [4차원 -> flow] -> concatenate

import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Conv1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



### ImageDataGenerator로 데이터 증폭시키기
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


# xy_train = imageGen.flow_from_directory(
#     '../data/real_age',
#     target_size=(32, 32),
#     batch_size=2000,
#     class_mode='categorical',
#     subset='training'
# )
# # Found 880 images belonging to 11 classes.
# ic(xy_train[0][0].shape)     #  (880, 32, 32, 3)
# ic(xy_train[0][1].shape)     #  (880, 11)


# xy_test = imageGen.flow_from_directory(
#     '../data/real_age',
#     target_size=(32, 32),
#     batch_size=2000,
#     class_mode='categorical',
#     subset='validation'
# )
# # Found 220 images belonging to 11 classes.
# ic(xy_test[0][0].shape)     # (220, 32, 32, 3)
# ic(xy_test[0][1].shape)     # (220, 11)



### 넘파이로 세이브하기
# np.save('./_save/_npy/proj_faceage_aug_x_train.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/proj_faceage_aug_x_test.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/proj_faceage_aug_y_train.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/proj_faceage_aug_y_test.npy', arr=xy_test[0][1])


# ============================================================================================================

### 데이터 로드하기
x_train = np.load('./_save/_npy/proj_faceage_aug_x_train.npy')
x_test = np.load('./_save/_npy/proj_faceage_aug_x_test.npy')
y_train = np.load('./_save/_npy/proj_faceage_aug_y_train.npy')
y_test = np.load('./_save/_npy/proj_faceage_aug_y_test.npy')

ic(x_train)
ic(x_test)
ic(y_train)
ic(y_test)
ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

'''
    x_train.shape: (880, 32, 32, 3)
    x_test.shape: (220, 32, 32, 3)
    y_train.shape: (880, 11)
    y_test.shape: (220, 11)
'''




#####랜덤
# 데이터 증폭
augment_size=50000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augment = x_train[randidx].copy()
y_augment = y_train[randidx].copy()
print('%%%%%%%%%%%%%%% 1 %%%%%%%%%%%%%%%%')
ic(x_augment.shape, y_augment.shape)        # (50000, 32, 32, 3), (50000, 11)


#####4차원
# x_augment = x_augment.reshape(x_augment.shape[0], 32, 32, 3)
# x_train = x_train_cifar10.reshape(x_train_cifar10.shape[0], 32, 32, 3)
# x_test = x_test_cifar10.reshape(x_test_cifar10.shape[0], 32, 32, 3)

#####flow
x_augment = imageGen.flow(# x와 y를 각각 불러옴
            x_augment,  # x
            np.zeros(augment_size),  # y
            batch_size=augment_size,
            save_to_dir='d:/temp/',
            shuffle=False).next()[0]
ic(type(x_augment), x_augment.shape)       # <class 'numpy.ndarray'>, (50000, 32, 32, 3)
print('%%%%%%%%%%%%%%% 2 %%%%%%%%%%%%%%%%')
ic(x_train.shape, x_augment.shape)  #(880, 32, 32, 3), (50000, 32, 32, 3)
ic(y_train.shape, y_augment.shape)  #(880, 11), (50000, 11)

#####concatenate
x_train = np.concatenate((x_train, x_augment))
y_train = np.concatenate((y_train, y_augment))
print('%%%%%%%%%%%%%%% 3 %%%%%%%%%%%%%%%%')
ic(x_train.shape, y_train.shape)        #  (50880, 32, 32, 3), (50880, 11)





ic(np.unique(y_train))  # 0, 1 : 2개
# 전처리 하기 -> scailing
# 단, 2차원 데이터만 가능하므로 4차원 -> 2차원
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)

# 1-2. x 데이터 전처리
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)



# 2. 모델 구성(GlobalAveragePooling2D 사용)
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same',                        
                        activation='relu' ,input_shape=(32, 32, 3)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(2,2))                                                     
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(2,2))                  
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(84, activation='relu'))
model.add(Dense(11, activation='softmax'))



# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/face_age_MCP2.hdf5')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, verbose=2, callbacks=[es, cp], validation_split=0.05, shuffle=True, batch_size=512)
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/face_age_model_save2.h5')

# model = load_model('./_save/ModelCheckPoint/face_age_model_save.h5')           # save model
# model = load_model('./_save/ModelCheckPoint/face_age_MCP.hdf5')                # checkpoint

# 4. 평가, 예측
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print("걸린시간 :", end_time)
print('acc :',acc[-1])
print('val_acc :',val_acc[-1])
print('val_loss :',val_loss[-1])


# 시각화 
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1) # 2개의 플롯을 할건데, 1행 1열을 사용하겠다는 의미 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2) # 2개의 플롯을 할건데, 1행 2열을 사용하겠다는 의미 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()


'''
*augment 전
걸린시간 : 6.861128091812134
acc : 0.40430623292922974
val_acc : 0.20454545319080353

*augment 50000
'./_save/ModelCheckPoint/face_age_model_save.h5'
걸린시간 : 113.05613613128662
acc : 0.8466567397117615
val_acc : 0.6780660152435303
val_loss : 1.1923494338989258
'''