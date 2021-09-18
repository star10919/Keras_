import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # DecisionTree의 앙상블 모델 : 숲(Foreset)
import warnings
warnings.filterwarnings('ignore')

# 실습
# 모델 : RandomForestClassifier

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

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV         # GridSearchCV : 체로 걸러서 찾겠다, CV(cross_val_score)까지 하겠다!!
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.995, shuffle=True, random_state=24)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)     # n_splits=5   5등분하겠다!  ->   값도 5(n)개로 나옴


parameters = [
    {'n_estimators':[100, 200], 'max_depth':[6, 8, 10, 12]},
    {'max_depth':[6, 8, 10, 12], 'min_samples_leaf':[3, 5, 7, 10]},
    {'min_samples_leaf':[3, 5, 7, 10], 'min_samples_split':[2, 3, 5, 10]},
    {'min_samples_split':[2, 3, 5, 10], 'n_jobs':[-1, 2, 4]},
    {'n_jobs':[-1, 2, 4], 'n_estimators':[100, 200]}
]



# 2. 모델 구성
from sklearn.ensemble import RandomForestClassifier     # DecisionTree의 앙상블 모델 : 숲(Foreset)
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 58 candidates, totalling 290 fits

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits


# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
from sklearn.metrics import accuracy_score

print("최적의 매개변수 :", model.best_estimator_)
print("best_score :", model.best_score_)



print("model.score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

'''
* GridSearchCV
최적의 매개변수 : RandomForestClassifier(n_jobs=4)
best_score : 0.685824882851577
model.score : 0.8
accuracy_score : 0.8

model.score : 0.8
accuracy_score : 0.8


* RandomizedSearchCV
최적의 매개변수 : RandomForestClassifier(n_jobs=-1)
best_score : 0.6864379508239878
model.score : 0.76
accuracy_score : 0.76
'''