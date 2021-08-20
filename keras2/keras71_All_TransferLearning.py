### pre-trained model

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

# model = Xception()
# model = ResNet50()
# model = ResNet50V2()
# model = ResNet101()
# model = ResNet101V2()
# model = ResNet152()
# model = ResNet152V2()
# model = DenseNet121()
# model = DenseNet169()
# model = DenseNet201()
model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = MobileNetV3Large()
# model = MobileNetV3Small()
# model = NASNetLarge()
# model = NASNetMobile()
# model = EfficientNetB0()
# model = EfficientNetB1()
# model = EfficientNetB7()



model.trainable=False

model.summary()

print("전체 가중치 갯수    :", len(model.weights))
print("훈련가능 가중치 갯수 :", len(model.trainable_weights))



# 모델별로 파라미터와 웨이트 수 정리

# <Xception>
# Total params: 22,910,480
# Trainable params: 0
# Non-trainable params: 22,910,480
# 236
# 0

# <ResNet50>
# Total params: 25,636,712
# Trainable params: 0
# Non-trainable params: 25,636,712
# 전체 가중치 갯수    : 320
# 훈련가능 가중치 갯수 : 0

# <ResNet50V2>
# Total params: 25,613,800
# Trainable params: 0
# Non-trainable params: 25,613,800
# 전체 가중치 갯수    : 272
# 훈련가능 가중치 갯수 : 0

# <ResNet101>
# Total params: 44,707,176
# Trainable params: 0
# Non-trainable params: 44,707,176
# 전체 가중치 갯수    : 626
# 훈련가능 가중치 갯수 : 0

# <ResNet101V2>
# Total params: 44,675,560
# Trainable params: 0
# Non-trainable params: 44,675,560
# 전체 가중치 갯수    : 544
# 훈련가능 가중치 갯수 : 0

# <ResNet152>
# Total params: 60,419,944
# Trainable params: 0
# Non-trainable params: 60,419,944
# 전체 가중치 갯수    : 932
# 훈련가능 가중치 갯수 : 0

# <ResNet152V2>
# Total params: 60,380,648
# Trainable params: 0
# Non-trainable params: 60,380,648
# 전체 가중치 갯수    : 816
# 훈련가능 가중치 갯수 : 0

# <DenseNet121>
# Total params: 8,062,504
# Trainable params: 0
# Non-trainable params: 8,062,504
# 전체 가중치 갯수    : 606
# 훈련가능 가중치 갯수 : 0

# <DenseNet169>
# Total params: 14,307,880
# Trainable params: 0
# Non-trainable params: 14,307,880
# 전체 가중치 갯수    : 846
# 훈련가능 가중치 갯수 : 0

# <DenseNet201>
# Total params: 20,242,984
# Trainable params: 0
# Non-trainable params: 20,242,984
# 전체 가중치 갯수    : 1006
# 훈련가능 가중치 갯수 : 0

# <InceptionV3>


# <InceptionResNetV2>


# <MobileNet>


# <MobileNetV2>


# <MobileNetV3Large>


# <MobileNetV3Small>


# <NASNetLarge>


# <NASNetMobile>


# <EfficientNetB0>


# <EfficientNetB1>


# <EfficientNetB7>
