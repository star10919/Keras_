import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  #, KNeighborsRegressor     # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier     # 의사결정나무
from sklearn.ensemble import RandomForestClassifier # DecisionTree의 앙상블 모델 : 숲(Foreset)
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',       # 경로잡기 중요!
                        index_col=None, header=0)    #header=0 첫번째라인   # (4898,12)


datasets_np = datasets.to_numpy()   #1 판다스 -> 넘파이
ic(datasets_np)
x = datasets_np[:,0:11]
# ic(x)
y = datasets_np[:,[-1]]
# ic(y)
# ic(x.shape, y.shape)   # x.shape: (4898, 11), y.shape: (4898,1)
# ic(np.unique(y))   # [3, 4, 5, 6, 7, 8, 9]  -  7개

from sklearn.model_selection import train_test_split, KFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.995, shuffle=True, random_state=24)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)     # n_splits=5   5등분하겠다!  ->   값도 5(n)개로 나옴


# 2. 모델 구성
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