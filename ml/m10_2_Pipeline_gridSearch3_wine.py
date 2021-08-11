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
    {'rf__n_estimators':[100, 200], 'rf__max_depth':[6, 8, 10, 12]},
    {'rf__max_depth':[6, 8, 10, 12], 'rf__min_samples_leaf':[3, 5, 7, 10]},
    {'rf__min_samples_leaf':[3, 5, 7, 10], 'rf__min_samples_split':[2, 3, 5, 10]},
    {'rf__min_samples_split':[2, 3, 5, 10], 'rf__n_jobs':[-1, 2, 4]},
    {'rf__n_jobs':[-1, 2, 4], 'rf__n_estimators':[100, 200]}
]


from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 2. 모델 구성
from sklearn.ensemble import RandomForestClassifier     # DecisionTree의 앙상블 모델 : 숲(Foreset)
from sklearn.pipeline import make_pipeline, Pipeline

# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([("scaler", MinMaxScaler()), ("rf", RandomForestClassifier())])


model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)


# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
print("최적의 매개변수 :", model.best_estimator_)   # best_estimator_, best_params_ 비슷
print("최적의 매개변수 :", model.best_params_)
print("best_score :", model.best_score_)


from sklearn.metrics import accuracy_score

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


* make_pipeline - 랜포사용
model.score : 0.76
accuracy_score : 0.76

* make_pipeline, GridSearchCV 사용
(best_estimator_)
최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(min_samples_split=3, n_jobs=4))])
(best_params_)
최적의 매개변수 : {'randomforestclassifier__min_samples_split': 3, 'randomforestclassifier__n_jobs': 4}
best_score : 0.6876725109250776
model.score : 0.76
accuracy_score : 0.76


* Pipeline
최적의 매개변수 : Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rf', RandomForestClassifier(n_estimators=200, n_jobs=4))])
최적의 매개변수 : {'rf__n_estimators': 200, 'rf__n_jobs': 4}
best_score : 0.685618069815195
model.score : 0.84
accuracy_score : 0.84
'''