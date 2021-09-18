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

from sklearn.model_selection import train_test_split, KFold, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)


#2. 모델구성(validation)
from sklearn.svm import LinearSVC, SVC      # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # 의사결정나무
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = LinearSVC()
# Acc : [0.         0.         0.01408451 0.         0.01428571]
# 평균 Acc : 0.0057

# model = SVC()
# Acc : [0.         0.01408451 0.         0.         0.01428571]
# 평균 Acc : 0.0057

# model = KNeighborsClassifier()
# Acc : [0. 0. 0. 0. 0.]
# 평균 Acc : 0.0

# model = KNeighborsRegressor()
# Acc : [0.50273145 0.42334975 0.20687826 0.39500877 0.31832704]
# 평균 Acc : 0.3693

# model = LogisticRegression()
# Acc : [0.         0.         0.         0.         0.01428571]
# 평균 Acc : 0.0029

# model = LinearRegression()
# Acc : [0.64364685 0.44506533 0.29292329 0.53733107 0.38844506]
# 평균 Acc : 0.4615

# model = DecisionTreeClassifier()
# Acc : [0.         0.         0.02816901 0.01428571 0.01428571]
# 평균 Acc : 0.0113

# model = DecisionTreeRegressor()
# Acc : [ 0.03729396 -0.11581662 -0.63727373  0.0233874  -0.25149635]
# 평균 Acc : -0.1888

# model = RandomForestClassifier()
# Acc : [0.         0.01408451 0.         0.         0.01428571]
# 평균 Acc : 0.0057

model = RandomForestRegressor()
# Acc : [0.48409463 0.3507672  0.24969061 0.41726665 0.37868407]
# 평균 Acc : 0.3761


#3. 컴파일(ES, reduce_lr), 훈련
#4. 평가, 예측(mse, r2)
scores = cross_val_score(model, x_train, y_train, cv=kfold)       #cross_val_score(모델, train과 test를 분리하지 않은 데이터, kfold)
print("Acc :", scores)      # 값이 n_splits의 개수로 나옴
print("평균 Acc :", round(np.mean(scores), 4))