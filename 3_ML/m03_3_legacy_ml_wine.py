import numpy as np
import pandas as pd
from icecream import ic
from tensorflow.python.keras.layers.recurrent import LSTM

### 다양한 레거시 머신러닝(evaluate -> score)
# 다중분류
# 모델링하고
# 0.8 이상 완성


# 1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',       # 경로잡기 중요!
                        index_col=None, header=0)    #header=0 첫번째라인   # (4898,12)

# * visual studio 기준(파이참이랑 다름)
    # ./  :  현재폴더(STUDY)
    # ../ :  상위폴더

# print(datasets)   # quality 가  y

# 아래 3개는 꼭 찍어보기
    # ic(datasets.shape)   # (4898, 12)   => x:(4898,11),   y:(4898,) 으로 잘라주기
    # ic(datasets.info())
    # ic(datasets.describe())


#1 판다스 -> 넘파이 : index와 header가 날아감
#2 x와 y를 분리
#3. sklearn의 OneHotEncoder 사용할것
#3 y의 라벨을 확인 np.unique(y) <= Output node 개수 잡아주기 위해서
#5. y의 shape 확인 (4898,) -> (4898,7)


datasets_np = datasets.to_numpy()   #1 판다스 -> 넘파이
ic(datasets_np)
x = datasets_np[:,0:11]
ic(x)
y = datasets_np[:,[-1]]
ic(y)
ic(x.shape, y.shape)   # x.shape: (4898, 11), y.shape: (4898,1)
ic(np.unique(y))   # [3, 4, 5, 6, 7, 8, 9]  -  7개


# from sklearn.preprocessing import OneHotEncoder    # 0, 1, 2 자동 채움 안됨 / # to_categorical 0, 1, 2 없으나 자동 생성
# onehot = OneHotEncoder()
# onehot.fit(y)
# y = onehot.transform(y).toarray() 
# ic(y.shape)    # (4898, 7)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.995, shuffle=True, random_state=24)

# x 데이터 전처리(scaler)
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   #  x_train.shape: (4873, 11), x_test.shape: (25, 11)
# x_train = x_train.reshape(4873, 11, 1)
# x_test = x_test.reshape(25, 11, 1)


# 2. 모델 구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Conv1D
# model = Sequential()
# model.add(LSTM(240, activation='relu', input_shape=(11,1), return_sequences=True))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(240, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(124, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(7, activation='softmax'))


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier      # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier         # 의사결정나무
from sklearn.ensemble import RandomForestClassifier     # DecisionTree의 앙상블 모델 : 숲(Foreset)

# model = LinearSVC()
# model.score : 0.6

# model = SVC()
# model.score : 0.64

# model = KNeighborsClassifier()
# model.score : 0.72

# model = LogisticRegression()
# model.score : 0.52

# model = DecisionTreeClassifier()
# model.score : 0.52

model = RandomForestClassifier()
# model.score : 0.88


# 3. 컴파일(ES, reduce_lr), 훈련
# from tensorflow.keras.optimizers import Adam, Nadam
# optimizer = Adam(lr=0.01)
# # optimizer = Nadam(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

# import time
# start = time.time()
model.fit(x_train, y_train)
# end = time.time() - start

# 4. 평가, 예측
# results = model.evaluate(x_test, y_test)
# print('걸린시간 :', end)
# print('category :', results[0])
# print('accuracy :', results[1])

# ic(y_test[-5:-1])
# y_predict = model.predict(x_test)
# ic(y_predict[-5:-1])

results = model.score(x_test, y_test)       # score 로 나오는 값 : accuracy_score
print("model.score :", results)

'''
category : 0.9682848453521729
accuracy : 0.800000011920929

*cnn + Flatten
걸린시간 : 13.947882652282715
category : 0.9115962386131287
accuracy : 0.6399999856948853

*cnn + GAP
걸린시간 : 29.681384325027466
category : 1.0343053340911865
accuracy : 0.4399999976158142

*LSTM
걸린시간 : 50.68032455444336
category : 0.8400266766548157
accuracy : 0.6000000238418579

*LSTM + Conv1D
걸린시간 : 62.63808012008667
category : 0.7918718457221985
accuracy : 0.6399999856948853

*reduce LR
걸린시간 : 40.95983624458313
category : 0.8441729545593262
accuracy : 0.6399999856948853
'''