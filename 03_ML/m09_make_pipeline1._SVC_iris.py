import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, r2_score

### 모델에 스케일링을 엮은 pipeline
# pipeline을 사용하여 scaling, modeling을 한번에

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=9)


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 2. 모델(머신러닝에서는 정의만 해주면 됨)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  #, KNeighborsRegressor     # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier     # 의사결정나무
from sklearn.ensemble import RandomForestClassifier # DecisionTree의 앙상블 모델 : 숲(Foreset)

from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), SVC())     # make_pipeline(스케일링, 모델)    # pipeline을 사용하여 scaling, modeling을 한번에





# 3. 훈련(컴파일 포함되어 있어서 컴파일 할 필요 없음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
print("model.score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

'''
model.score : 1.0
accuracy_score : 1.0
'''

