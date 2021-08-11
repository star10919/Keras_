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
# Acc : [0.31692308 0.44717949 0.34564103 0.17453799 0.43737166]
# 평균 Acc : 0.3443

# model = SVC()
# Acc : [0.44410256 0.44923077 0.45128205 0.47433265 0.42299795]
# 평균 Acc : 0.4484

# model = KNeighborsClassifier()
# Acc : [0.46358974 0.47487179 0.48205128 0.48151951 0.48459959]
# 평균 Acc : 0.4773

# model = LogisticRegression()
# Acc : [0.47487179 0.46769231 0.46051282 0.46201232 0.4650924 ]
# 평균 Acc : 0.466

# model = DecisionTreeClassifier()
# Acc : [0.59179487 0.57435897 0.59384615 0.61396304 0.64271047]
# 평균 Acc : 0.6033

model = RandomForestClassifier()
# Acc : [0.69333333 0.66974359 0.66666667 0.6889117  0.69609856]
# 평균 Acc : 0.683


# 3. 컴파일(ES, reduce_lr), 훈련
# 4. 평가(evaluate 대신 score 사용함!!), 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)       #cross_val_score(모델, train과 test를 분리하지 않은 데이터, kfold)
print("Acc :", scores)      # 값이 n_splits의 개수로 나옴
print("평균 Acc :", round(np.mean(scores),4))