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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


### make_pipeline도 모델임


#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

parameters = [ # pipeline에서 사용하는 모델명을 파라미터 앞에 적어주면 사용할 수 있음   /  모델명__파라미터명(모델명, 파라미터 모두 소문자로 써야함, 모델명과 파라미터 구분은 __(언더바 2개로))
    {'randomforestclassifier__min_samples_leaf':[3, 5, 7], 'randomforestclassifier__max_depth':[2, 3, 5, 10]},
    {'randomforestclassifier__min_samples_split':[6, 8, 10]}

]# @@@2 파라미터 앞에 사용한 파라미터의 모델명(랜포) 소문자로 적어주기

#2. 모델구성(validation)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

# @@@1 make_pipeline(pipe) 라는 모델 사용하고 있는데, 파라미터는 랜포 파라미터라서 에러남
model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)


# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
print("최적의 매개변수 :", model.best_estimator_)   # best_estimator_, best_params_ 비슷
print("최적의 매개변수 :", model.best_params_)
print("best_score :", model.best_score_)



print("model.score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("r2_score :", accuracy_score(y_test, y_predict))

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
'''