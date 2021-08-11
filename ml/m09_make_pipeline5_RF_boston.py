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

from sklearn.preprocessing import MinMaxScaler


#2. 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), RandomForestRegressor())

# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
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
'''