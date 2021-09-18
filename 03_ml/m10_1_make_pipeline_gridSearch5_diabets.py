import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # DecisionTree의 앙상블 모델 : 숲(Foreset)
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score


# 실습
# 모델 : RandomForestClassifier

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

parameters = [{
    "randomforestregressor__max_depth": [6, 8, 10, 12],
    "randomforestregressor__min_samples_leaf": [3, 5, 7],
    "randomforestregressor__min_samples_split": [2, 3, 5, 10],
}]



#2. 모델구성(validation)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())

model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)


# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
print("최적의 매개변수 :", model.best_estimator_)   # best_estimator_, best_params_ 비슷
print("최적의 매개변수 :", model.best_params_)
print("best_score :", model.best_score_)


print("model.score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("r2_score :", r2_score(y_test, y_predict))

'''
* GridSearchCV
최적의 매개변수 : RandomForestRegressor(min_samples_leaf=10, min_samples_split=3)
best_score : 0.4172036841176213
model.score : 0.5503341505809245
r2_score : 0.5503341505809245

model.score : 0.5346895000833456
r2_score : 0.5346895000833456


* RandomizedSearchCV
최적의 매개변수 : RandomForestRegressor(min_samples_leaf=5)
best_score : 0.39637705062401213
model.score : 0.5534191234527281
r2_score : 0.5534191234527281


* make_pipeline - 랜포사용
model.score : 0.5265411480154842
r2_score : 0.5265411480154842


* make_pipeline, GridSearchCV 사용
(best_estimator_)
최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestregressor',
                 RandomForestRegressor(max_depth=6, min_samples_leaf=7))])
(best_params_)
최적의 매개변수 : {'randomforestregressor__max_depth': 6, 'randomforestregressor__min_samples_leaf': 7, 'randomforestregressor__min_samples_split': 2}
best_score : 0.4159586518036578
model.score : 0.5547792214520815
r2_score : 0.5547792214520815
'''