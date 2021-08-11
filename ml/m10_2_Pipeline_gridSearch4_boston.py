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

parameters = [{
    "rf__max_depth": [6, 8, 10, 12],
    "rf__min_samples_leaf": [3, 5, 7],
    "rf__min_samples_split": [2, 3, 5, 10],
}]

from sklearn.preprocessing import MinMaxScaler


#2. 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())
pipe = Pipeline([("scaler", MinMaxScaler()), ("rf", RandomForestRegressor())])


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


* make_pipeline - 랜포사용
model.score : 0.8934027481631501
r2_score : 0.8934027481631501


* make_pipeline, GridSearchCV 사용
(best_estimator_)
최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestregressor',
                 RandomForestRegressor(max_depth=12, min_samples_leaf=3,
                                       min_samples_split=3))])
(best_params_)
최적의 매개변수 : {'randomforestregressor__max_depth': 12, 'randomforestregressor__min_samples_leaf': 3, 'randomforestregressor__min_samples_split': 3}
best_score : 0.8238253814988138
model.score : 0.8819587840620349
r2_score : 0.8819587840620349


* Pipeline
최적의 매개변수 : Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rf',
                 RandomForestRegressor(max_depth=6, min_samples_leaf=3,
                                       min_samples_split=3))])
최적의 매개변수 : {'rf__max_depth': 6, 'rf__min_samples_leaf': 3, 'rf__min_samples_split': 3}
best_score : 0.8233058146036525
model.score : 0.8830012203414173
r2_score : 0.8830012203414173
'''