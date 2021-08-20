from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19     # layer 깊이가 16, 19

model = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))   # include_top=False : input_shape 조정 가능
# model = VGG16()
# model = VGG19()

model.trainable=False   # False : imagenet의 가중치를 그대로 가져다 쓰겠다!(훈련하지 않겠다=가중치의 갱신이 없다)

model.summary()
                     # model.trainable=True(default)     False
print(len(model.weights))               # 26               26
print(len(model.trainable_weights))     # 26                0


'''
* default
<VGG16>
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0

<VGG19>
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
'''

# _________________________________________________________________        
# Layer (type)                 Output Shape              Param #
# =================================================================        
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0              (224, 224, 3) : default
# _________________________________________________________________        
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# ...........................
# ...........................
# _________________________________________________________________        
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________        
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________        
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________        
# predictions (Dense)          (None, 1000)              4097000
# =================================================================  
# Total params: 143,667,240
# Trainable params: 143,667,240
# Non-trainable params: 0 

# FC(fully connected) <- 용어정리하기


# * model.trainable=False
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688