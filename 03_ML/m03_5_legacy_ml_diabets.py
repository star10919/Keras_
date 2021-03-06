from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

### 다양한 레거시 머신러닝(evaluate -> score)
# 실습, 모델구성하고 완료하시오.
# 회귀 데이터를 Classifier로 만들었을 경우에 에러 확인!!!

# 클래시파이(분류모델)가 아니므로 에러가 날거임
# 리그레서(회귀모델) 임!!!!


#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # (442, 10)  (442,)

# ic(datasets.feature_names)   
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

# ic(x[:30])
# ic(np.min(y), np.max(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

# x 데이터 전처리
from sklearn.preprocessing import StandardScaler, PowerTransformer, MaxAbsScaler
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   # x_train.shape: (353, 10), x_test.shape: (89, 10)

# x_train = x_train.reshape(353, 10, 1)
# x_test = x_test.reshape(89, 10, 1)

#2. 모델구성(validation)
from sklearn.svm import LinearSVC, SVC      # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # 의사결정나무
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = LinearSVC()
# model.score : 0.011235955056179775

# model = SVC()
# model.score : 0.011235955056179775

# model = KNeighborsClassifier()
# model.score : 0.011235955056179775

# model = KNeighborsRegress

# model = LogisticRegression()
# model.score : 0.011235955056179775

# model = LinearRegression()
# model.score : 0.5851141269959736

# model = DecisionTreeClassifier()
# model.score : 0.011235955056179775

# model = DecisionTreeRegressor()
# model.score : -0.1285518296551733

# model = RandomForestClassifier()
# model.score : 0.0

model = RandomForestRegressor()
# model.score : 0.5460313800399677








# model = Sequential()
# model.add(LSTM(10, input_shape=(10, 1), activation='relu', return_sequences=True))
# model.add(Conv1D(100, 2, activation='relu'))  #relu : 음수값은 0, 양수만 제대로 잡음
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))


#3. 컴파일(ES, reduce_lr), 훈련
# from tensorflow.keras.optimizers import Adam, Nadam
# optimizer = Adam(lr=0.01)
# # optimizer = Nadam(lr=0.01)
# model.compile(loss='mse', optimizer=optimizer)

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# es = EarlyStopping(monitor='val_loss', mode='min', patience=16)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=9, mode='auto', verbose=1, factor=0.1)

# import time
# start = time.time()
model.fit(x_train, y_train)
# end = time.time() - start

# #4. 평가, 예측(mse, r2)

results = model.score(x_test, y_test)       # score 로 나오는 값 : accuracy_score
print("model.score :", results)

# loss = model.evaluate(x_test, y_test)
# print('걸린시간 :', end)
# ic(loss)

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# ic(r2)



'''
* MaxAbsScaler
ic| loss: 2240.12841796875
ic| r2: 0.588361118619634

*cnn + Flatten
ic| loss: 2341.884765625
ic| r2: 0.5696626833347622

*cnn + GAP
ic| loss: 3888.330810546875
ic| r2: 0.2854926965183139

*LSTM
걸린시간 : 88.2120795249939
ic| loss: 2421.381591796875
ic| r2: 0.5550545863459289

*Conv1D
걸린시간 : 16.57935118675232
ic| loss: 2188.756591796875
ic| r2: 0.5978010110126091

*LSTM + Conv1D
걸린시간 : 91.52887654304504
ic| loss: 2080.7890625
ic| r2: 0.6176408093415701

*reduce LR
걸린시간 : 23.658804655075073
ic| loss: 2077.2978515625
ic| r2: 0.6182823199819709
'''