from tensorflow.keras.datasets import fashion_mnist
import numpy as np

### flow

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

## ImageDataGenerator로 데이터 증폭시키기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest',
    )

#1. ImageDataGenerator를 정의
#2. 파일에서 땡겨오려면 -> flow_from_directory()  :  xy 가 튜플형태로 묶여서 나옴
#3. 데이터에서 땡겨오려면 -> flow()  :  x와 y가 분류되어 있어야 한다.

argument_size=40000

randidx = np.random.randint(x_train.shape[0], size=argument_size)       # x_train[0]에서 아크먼트 사이즈 만큼 랜덤하게 들어감
print(x_train.shape[0])     # 60000
print(randidx)              # [44596 49164  1092 ... 51768  3501 13118]
print(randidx.shape)        # (40000,)

x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()
print(x_argumented.shape)       # (40000, 28, 28)

# flow 의 x는 4차원을 받아야 한다!!
x_argumented = x_argumented.reshape(x_argumented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

                                    # x        # y (y_argumented 넣어도 됨)
x_argumented = train_datagen.flow(x_argumented, np.zeros(argument_size), batch_size=argument_size, shuffle=False).next()[0]   # x만 추출([0])      # .next() 써주기
print(x_argumented.shape)       # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_argumented))
y_train = np.concatenate((y_train, y_argumented))

print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)