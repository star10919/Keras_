import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from icecream import ic
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

# 실습
# 모델 : RandomForestClassifier

# 1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',       # 경로잡기 중요!
                        index_col=None, header=0)    #header=0 첫번째라인   # (4898,12)

datasets_np = datasets.to_numpy()   #1 판다스 -> 넘파이
x = datasets_np[:,0:11]
y = datasets_np[:,[-1]]


from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV         # GridSearchCV : 체로 걸러서 찾겠다, CV(cross_val_score)까지 하겠다!!
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.995, shuffle=True, random_state=24)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)     # n_splits=5   5등분하겠다!  ->   값도 5(n)개로 나옴

parameters = [
    {'randomforestclassifier__n_estimators':[100, 200]},
    {'randomforestclassifier__max_depth':[6, 8, 10, 12]},
    {'randomforestclassifier__min_samples_leaf':[3, 5, 7, 10]},
    {'randomforestclassifier__min_samples_split':[2, 3, 5, 10]},
    {'randomforestclassifier__n_jobs':[-1, 2, 4]}
]

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 2. 모델 구성
from sklearn.ensemble import RandomForestClassifier     # DecisionTree의 앙상블 모델 : 숲(Foreset)
from sklearn.pipeline import make_pipeline, Pipeline
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)


# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)

# 4. 평가(evaluate 대신 score 사용함!!), 예측
print("최적의 매개변수 :", model.best_estimator_)
print("최적의 매개변수 :", model.best_params_)
print("best_score :", model.best_score_)



print("model.score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

'''
* GridSearchCV
Fitting 5 folds for each of 17 candidates, totalling 85 fits
최적의 매개변수 : RandomForestClassifier()
best_score : 0.6845926393934608
model.score : 0.8
accuracy_score : 0.8


* RandomizedSearchCV
최적의 매개변수 : RandomForestClassifier()
best_score : 0.682541146738272
model.score : 0.8
accuracy_score : 0.8


* make_pipeline - 랜포사용
(best_estimator_)
최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier', RandomForestClassifier(n_jobs=4))])

(best_params_)
최적의 매개변수 : {'randomforestclassifier__n_jobs': 4}
best_score : 0.6856189122308219
model.score : 0.76
accuracy_score : 0.76
'''