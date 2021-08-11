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

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV         # GridSearchCV : 체로 걸러서 찾겠다, CV(cross_val_score)까지 하겠다!!
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 2. 모델(머신러닝에서는 정의만 해주면 됨)  GridSearchCV로 모델(SVC) 감싸줌
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  #, KNeighborsRegressor     # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier     # 의사결정나무
from sklearn.ensemble import RandomForestClassifier # DecisionTree의 앙상블 모델 : 숲(Foreset)

from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), RandomForestClassifier())     # make_pipeline(스케일링, 모델)    # pipeline을 사용하여 scaling, modeling을 한번에



# 3. 훈련(cross_val_score 은 fit과 score가 포함되어 있음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
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

* make_pipeline - RandomForestClassifier 사용
model.score : 0.9111111111111111
accuracy_score : 0.9111111111111111
'''