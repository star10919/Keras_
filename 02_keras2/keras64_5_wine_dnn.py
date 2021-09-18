import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_wine
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from icecream import ic
import warnings
warnings.filterwarnings('ignore')

### 다중 분류 (0,1,2를 명목척도로 바꿔줘야 함 <- 원핫인코딩 사용(to_categorical))
### Output activation = 'softmax', 컴파일 loss = 'categorical_crossentropy'

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)        # ic| x.shape: (178, 13), y.shape: (178,)
ic(y)   # (0,0,0, ... ,1,1,1 ...)
ic(np.unique(y))    # 0,1,2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# 2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(13,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

def create_hyperparameter():
    batches = [1000, 2000, 3000, 4000, 5000]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    return {"batch_size" : batches, "optimizer": optimizer, "drop" : dropout}


hyperparameters = create_hyperparameter()
# print(hyperparameters)      # {'batch_size': [10, 20, 30, 40, 50], 'optimizer': ['rmsprop', 'adam', 'adadelta'], 'drop': [0.1, 0.2, 0.3]}
# model2 = buile_model()


###############################################################################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  # 사이킷런으로 케라스(텐서플로우)모델 감쌈
model2 = KerasClassifier(build_fn=build_model, verbose=1, validation_split=0.2) #, epochs=2)  # ***epochs는 2군데 먹힘-1, validation_split도 2군데 먹힘
###############################################################################################################################


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = RandomizedSearchCV(model2, hyperparameters, cv=5)        # (모델, 하파튜, 크로스발리데이션) 
                            #텐서플로우 모델이라 에러뜸 -> 사이킷런으로 텐서플로우 모델 래핑해야 함!
model = GridSearchCV(model2, hyperparameters, cv=2)   

model.fit(x_train, y_train, verbose=1, epochs=3)#, validation_split=0.2)  # ***epochs는 2군데 먹힘-2  /  둘이 같이 주면 아래꺼가 먹힘 /  validation_split도 2군데 먹힘

print(model.best_estimator_)
print(model.best_params_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 :", acc)

'''
{'batch_size': 1000, 'drop': 0.4, 'optimizer': 'adam'}
최종 스코어 : 0.46296295523643494
'''