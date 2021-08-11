import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  #, KNeighborsRegressor     # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier     # 의사결정나무
from sklearn.ensemble import RandomForestClassifier # DecisionTree의 앙상블 모델 : 숲(Foreset)
import warnings
warnings.filterwarnings('ignore')

### kfold : test 데이터는 훈련 못시키니까   발생하는 train 데이터 부족현상 해결
# (train_test_split 불필요)

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)

from sklearn.model_selection import train_test_split, KFold, cross_val_score         # cross_val_score : 교차검증(KFold와 유사)
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)     # n_splits=5   5등분하겠다!  ->   값도 5(n)개로 나옴


# 2. 모델(머신러닝에서는 정의만 해주면 됨)
# model = LinearSVC()
# Acc : [0.96666667 0.96666667 1.         0.9        1.        ]
# 평균 Acc : 0.9667

# model = SVC()
# Acc : [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# 평균 Acc : 0.9667

# model = KNeighborsClassifier()
# Acc : [0.96666667 0.96666667 1.         0.9        0.96666667]
# 평균 Acc : 0.96

model = LogisticRegression()
# Acc : [1.         0.96666667 1.         0.9        0.96666667]
# 평균 Acc : 0.9667

# model = DecisionTreeClassifier()
# Acc : [0.96666667 0.96666667 1.         0.9        0.93333333]
# 평균 Acc : 0.9533

# model = RandomForestClassifier()
# Acc : [0.93333333 0.96666667 1.         0.9        0.96666667]
# 평균 Acc : 0.9533


# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
# 4. 평가(evaluate 대신 score 사용함!!), 예측
scores = cross_val_score(model, x, y, cv=kfold)       #cross_val_score(모델, train과 test를 분리하지 않은 데이터, kfold)
print("Acc :", scores)      # 값이 n_splits의 개수로 나옴
print("평균 Acc :", round(np.mean(scores),4))


