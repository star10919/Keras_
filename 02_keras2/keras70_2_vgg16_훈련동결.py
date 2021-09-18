from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.python.keras.applications import vgg16     # layer 깊이가 16, 19

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))   # include_top=False : input_shape 조정 가능

vgg16.trainable=False   # vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))        # *layer 1 추가
model.add(Dense(1))         # *layer 2 추가

# model.trainable=False   # 전체 모델 훈련을 동결한다.(True가 default)

model.summary()

print(len(model.weights))               # 26 -> 30(layer 2개 추가 : 2(w+b)=4)
print(len(model.trainable_weights))     # 0 -> 4
