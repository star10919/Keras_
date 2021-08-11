import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

# 실습
# 모델 : RandomForestClassifier

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV         # GridSearchCV : 체로 걸러서 찾겠다, CV(cross_val_score)까지 하겠다!!
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)     # n_splits=5   5등분하겠다!  ->   값도 5(n)개로 나옴


parameters = [
    {'n_estimators':[100, 200]},
    {'max_depth':[6, 8, 10, 12]},
    {'min_samples_leaf':[3, 5, 7, 10]},
    {'min_samples_split':[2, 3, 5, 10]},
    {'n_jobs':[-1, 2, 4]}
]


# 2. 모델(머신러닝에서는 정의만 해주면 됨)  GridSearchCV로 모델(SVC) 감싸줌
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 17 candidates, totalling 85 fits

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits


# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
print("최적의 매개변수 :", model.best_estimator_)
print("best_score :", model.best_score_)



print("model.score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

'''
* GridSearchCV
최적의 매개변수 : RandomForestClassifier(min_samples_leaf=10)
best_score : 0.9714285714285713
model.score : 0.9333333333333333
accuracy_score : 0.9333333333333333

model.score : 0.9333333333333333
accuracy_score : 0.9333333333333333


* RandomizedSearchCV
최적의 매개변수 : RandomForestClassifier(n_jobs=4)
best_score : 0.9619047619047618
model.score : 0.9111111111111111
accuracy_score : 0.9111111111111111
'''