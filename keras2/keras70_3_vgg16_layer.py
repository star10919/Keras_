from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.python.keras.applications import vgg16     # layer 깊이가 16, 19

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))   # include_top=False : input_shape 조정 가능

vgg16.trainable=False   # vgg훈련을 동결한다

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))        # *layer 1 추가
model.add(Dense(1))         # *layer 2 추가

# model.trainable=False   # 전체 모델 훈련을 동결한다.

model.summary()

print(len(model.weights))               # 26 -> 30(layer 2개 추가 : 2(w+b)=4)
print(len(model.trainable_weights))     # 0 -> 4


###################### 2번 파일에서 아래만 추가 #####################
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns= ['Layer Type', 'Layer Name', 'Layer Trainable'])

print(results)

'''
                                                                            Layer Type  ... Layer Trainable
0  <tensorflow.python.keras.engine.functional.Functional object at 0x00000235FF9AEBB0>  ...  False
1  <tensorflow.python.keras.layers.core.Flatten object at 0x000002358B2B5C70>           ...  True
2  <tensorflow.python.keras.layers.core.Dense object at 0x000002358B2AAC70>             ...  True
3  <tensorflow.python.keras.layers.core.Dense object at 0x000002358B319820>             ...  True
'''