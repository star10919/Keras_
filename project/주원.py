import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# imggen = ImageDataGenerator(
#             rescale=1./255,
#             horizontal_flip=True,
#             vertical_flip=True,
#             width_shift_range=0.1,
#             height_shift_range=0.1,
#             rotation_range=5,
#             shear_range=0.5,
#             fill_mode='nearest',
#             validation_split=0.2)

pred_datagen = ImageDataGenerator(
              rescale=1./255,   
              horizontal_flip=True,
              vertical_flip=True,
              width_shift_range=0.1,
              height_shift_range=0.1,
              rotation_range=5,
              shear_range=0.5,
              fill_mode='nearest')

# test_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = imggen.flow_from_directory(
#             './_crawl_data',
#             target_size=(64,64),
#             batch_size=3800,
#             class_mode='categorical',
#             shuffle=True,
#             subset='training')

# # Found 3784 images belonging to 20 classes.

# xy_test = imggen.flow_from_directory(
#             './_crawl_data',
#             target_size=(64,64),
#             batch_size=1000,
#             shuffle=True,
#             class_mode='categorical',
#             subset='validation')

# # Found 935 images belonging to 20 classes.

result = ['angelonia flower', 'begonia flower', 'cape jasmine flower','chinese trumpet creeper flower','coreopsis flower',

            'geranium flower','gooseneck loosestrife flower','gypsophilia flower','marigold flower','moonbeam flower',

            'oenothera speciosa nutt flower','pansy flower','peony flower','petunia flower','Phlox Subulata flower',

            'porcupine flower','rose mallow flower','scotch broom flower','shasta daisy flower','silene armeria flower']

def flowername():
    while 1:
        start = input('s: 시작, 0: 종료\n입력: ')
        
        if start == '0':
            print('프로그램을 종료 합니다.')
            break
        elif start == 's':
            menu = int(input('숫자입력: '))
            return str(result[menu-1])           
        else:
            print('잘못된 선택 입니다.')
            continue

flower = flowername()
ic(꽃)
x_prd = pred_datagen.flow_from_directory('./_pred_data/'+ f'{flower}/' ,
              target_size=(100,100), batch_size=1)

x_train = np.load('./_save/_npy/project_train_x.npy') # 53774, 64, 64, 3
x_test = np.load('./_save/_npy/project_test_x.npy') # (933, 64, 64, 3)
y_train = np.load('./_save/_npy/project_train_y.npy') # (3784, 20)
y_test = np.load('./_save/_npy/project_test_y.npy') # (935, 20)


# ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # x_train.shape: (33784, 150, 150, 3), y_train.shape: (33784, 20)


# model = Sequential()
# model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(64, 64, 3))) 
# model.add(MaxPool2D(2,2))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
# model.add(MaxPool2D(2,2))               
# model.add(Dropout(0.3))
# model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D())
# # model.add(Flatten())                                              
# # model.add(Dense(128, activation='relu'))
# # model.add(Dense(512, activation='relu'))
# model.add(Dense(20, activation='softmax'))

# model.summary()

model = load_model('./_save/ModelCheckPoint/project0803_1126_.0032-0.472263.hdf5')


#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_prd)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(y_pred)
y_prd = np.argmax(y_pred, 1)
ic(y_prd[0])
# ic(f'{걸린시간}분')



for i in np.argmax(y_pred, 1):
    if i == 0 :
        print('기대하는 꽃은', f'{flower}이며 예측한 꽃은',result[0],"입니다")
    elif i == 1 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[1],"입니다")
    elif i == 2 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[2],"입니다")
    elif i == 3 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[3],"입니다")
    elif i == 4 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[4],"입니다")
    elif i == 5 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[5],"입니다")
    elif i == 6 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[6],"입니다")
    elif i == 7 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[7],"입니다")
    elif i == 8 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[8],"입니다")   
    elif i == 9 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[9],"입니다")
    elif i == 10 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[10],"입니다")
    elif i == 11 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[11],"입니다")
    elif i == 12 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[12],"입니다")
    elif i == 13 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[13],"입니다")
    elif i == 14 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[14],"입니다")
    elif i == 15 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[15],"입니다")
    elif i == 16 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[16],"입니다")
    elif i == 17 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[17],"입니다")
    elif i == 18 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[18],"입니다")
    elif i == 19 :
        print('기대하는 꽃은:', f'{flower} 이며 예측한 꽃은',result[19],"입니다")


# plt.figure(figsize=(9,5))
# plt.subplot(2, 1, 1) # 2개의 플롯을 할건데, 1행 1열을 사용하겠다는 의미
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# # 2
# plt.subplot(2, 1, 2) # 2개의 플롯을 할건데, 1행 2열을 사용하겠다는 의미 
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()

'''
scaler = Robust


'''
