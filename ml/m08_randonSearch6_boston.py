import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.ensemble import RandomForestRegressor # DecisionTree의 앙상블 모델 : 숲(Foreset)
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score

# 실습
# 모델 : RandomForestRegressor

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)


parameters = [
    {'n_estimators':[100, 200], 'max_depth':[6, 8, 10, 12], 'min_samples_leaf':[3, 5, 7, 10]},
    {'max_depth':[6, 8, 10, 12]},
    {'min_samples_leaf':[3, 5, 7, 10], 'min_samples_split':[2, 3, 5, 10]},
    {'min_samples_split':[2, 3, 5, 10]},
    {'n_jobs':[-1, 2, 4], 'max_depth':[6, 8, 10, 12]}
]


#2. 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 68 candidates, totalling 340 fits

model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits


# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
print("최적의 매개변수 :", model.best_estimator_)
print("best_score :", model.best_score_)



print("model.score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("r2_score :", r2_score(y_test, y_predict))

'''
* GridSearchCV
최적의 매개변수 : RandomForestRegressor(max_depth=8, n_jobs=4)
best_score : 0.8516834086835677
model.score : 0.8909888322610514
r2_score : 0.8909888322610514

model.score : 0.8822352783561465
r2_score : 0.8822352783561465


* RandomizedSearchCV
최적의 매개변수 : RandomForestRegressor(max_depth=10)
best_score : 0.8463887813739799
model.score : 0.890466695053099
r2_score : 0.890466695053099
'''