import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
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

# 보스톤은 회귀모델

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split, KFold, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)


#2. 모델

from sklearn.svm import LinearSVC, SVC      # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # 의사결정나무
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = LinearSVC()
# Acc : [nan nan nan nan nan]
# 평균 Acc : nan

# model = SVC()
# Acc : [nan nan nan nan nan]
# 평균 Acc : nan

# model = KNeighborsClassifier()
# Acc : [nan nan nan nan nan]
# 평균 Acc : nan

# model = KNeighborsRegressor()
# Acc : [0.48564217 0.44681511 0.58468507 0.2899459  0.49101289]
# 평균 Acc : 0.4596

# model = LogisticRegression()
# Acc : [nan nan nan nan nan]
# 평균 Acc : nan

# model = LinearRegression()
# Acc : [0.64520963 0.75944241 0.66980083 0.63214666 0.68836076]
# 평균 Acc : 0.679

# model = DecisionTreeClassifier()
# Acc : [nan nan nan nan nan]
# 평균 Acc : nan

# model = DecisionTreeRegressor()
# Acc : [0.76689555 0.81514083 0.59224788 0.80524831 0.77157417]
# 평균 Acc : 0.7502

# model = RandomForestClassifier()
# Acc : [nan nan nan nan nan]
# 평균 Acc : nan

model = RandomForestRegressor()
# Acc : [0.82794787 0.88481037 0.74833324 0.86579187 0.88290763]
# 평균 Acc : 0.842


#3. 컴파일(ES, reduce_lr), 훈련
#4. 평가, 예측, r2결정계수
scores = cross_val_score(model, x_train, y_train, cv=kfold)       #cross_val_score(모델, train과 test를 분리하지 않은 데이터, kfold)
print("Acc :", scores)      # 값이 n_splits의 개수로 나옴
print("평균 Acc :", round(np.mean(scores), 4))