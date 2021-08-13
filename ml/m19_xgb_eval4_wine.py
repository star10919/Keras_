from sklearn import datasets
from  xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from icecream import ic

### eval_set, eval_metric
# 분류 eval_metric=['merror', 'mlogloss']

# 1. 데이터
datasets = load_wine()
x = datasets.data       #datasets['data'] 도 가능
y = datasets.target

ic(x.shape, y.shape)    # (506, 13), (506,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 1-2. 데이터 전처리
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# 2. 모델
model = XGBClassifier(n_estimators=50, learing_rate=0.01, n_jobs=1)        # n_estimators = epochs
                    # (xgboost에서 알아야 할 파라미터들)

# 3. 훈련
model.fit(x_train, y_train, verbose=1,                           # verbose=1 : eval_set 보여줌
         eval_set=[(x_train, y_train), (x_test, y_test)],         # eval_set=[(훈련set, 검증set)] : 훈련되는거 보여줌
         eval_metric=['merror', 'mlogloss']
)


# 4. 평가, 예측
results = model.score(x_test, y_test)
ic(results)

y_predict = model.predict(x_test)
r2 = accuracy_score(y_test, y_predict)
ic(r2)

'''
ic| results: 1.0
ic| r2: 1.0
'''