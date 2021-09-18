from tensorflow.keras.datasets import fashion_mnist
import numpy as np

### flow  / 한개의 이미지를 여러개로 늘린 것

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


# xy_train = train_datagen.flow_from_directory(# x와 y가 동시에 생성됨
#     '../data/brain/train',      # 이미지가 있는 폴더 지정이 아닌 상위 폴더(동일한 급의 라벨이 모여있는 폴더까지)로 지정   # train(ad/normal)
#     target_size=(150, 150),     # 임의대로 크기 조정 - 해상도
#     batch_size=5,               # y 하나의 개수
#     class_mode='binary',        # 이상이 있다-라벨 / 이상이 없다-라벨 : 이진분류
#     shuffle=False
# )


#1. ImageDataGenerator를 정의
#2. 파일에서 땡겨오려면 -> flow_from_directory()  :  xy 가 튜플형태로 묶여서 나옴
#3. 데이터에서 땡겨오려면 -> flow()  :  x와 y가 분류되어 있어야 한다.



# 데이터 증폭
'''
배열을 반복하면서 새로운 축을 추가하기 : np.tile
np.tile(arr, reps) method 는 'arr' 에는 배열을, 'reps'에는 반복하고자 하는 회수를 넣어줍니다.
'reps'에는 숫자를 넣을 수도 있고, 배열을 넣을 수도 있습니다.
'''
augment_size=100
x_data = train_datagen.flow(# x와 y를 각각 불러옴
            np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),      # x      # x_train[0]을 아규먼트 사이즈 만큼 다른 각도로 복제함(증폭)
            np.zeros(augment_size),       # y
            batch_size=augment_size,
            shuffle=False
).next()   # => iterator 방식(.next()를 안 붙이면 iterator의 한 부분씩만 실행)으로 반환!!       //      # .next()를 붙이면 전체실행

# print(type(x_data))     # .next() x :<class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>   # Iterator : 반복자
#                         # .next() o :  -> <class 'tuple'>
# print(type(x_data[0]))      # <class 'tuple'>
#                         #   ->  <class 'numpy.ndarray'>
# print(type(x_data[0][0]))      # <class 'numpy.ndarray'>
# print(type(x_data[0][1]))      # <class 'numpy.ndarray'>
# print(x_data[0][0].shape)      # (100,28,28,1)
# print(x_data[0][1].shape)      # (100,)


import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()