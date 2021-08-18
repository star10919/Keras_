import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D

### 사이킷런모델(그리드서치,랜덤서치) 안에     케라스모델(텐서플로우) 사용하려면      KerasClassifier,KerasRegressor로     케라스모델 감싸줘야 함.

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255


# 2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer": optimizer, "drop" : dropout}


hyperparameters = create_hyperparameter()
# print(hyperparameters)      # {'batch_size': [10, 20, 30, 40, 50], 'optimizer': ['rmsprop', 'adam', 'adadelta'], 'drop': [0.1, 0.2, 0.3]}
# model2 = buile_model()


###############################################################################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  # 사이킷런으로 케라스(텐서플로우)모델 감쌈
model2 = KerasClassifier(build_fn=build_model, verbose=1)
###############################################################################################################################


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = RandomizedSearchCV(model2, hyperparameters, cv=5)        # (모델, 하파튜, 크로스발리데이션)
                            #텐서플로우 모델이라 에러뜸 -> 사이킷런으로 텐서플로우 모델 래핑해야 함!

model.fit(x_train, y_train, verbose=1)