from sklearn import datasets
from  xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from icecream import ic

### eval_set, eval_metric
# 회귀 eval_metric=['rmse', 'mae', 'logloss'],  default : rmse

# 1. 데이터
datasets = load_boston()
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
model = XGBRegressor(n_estimators=20, learing_rate=0.05, n_jobs=1)        # n_estimators = epochs
                    # (xgboost에서 알아야 할 파라미터들)

# 3. 훈련
model.fit(x_train, y_train, verbose=1,                           # verbose=1 : eval_set 보여줌
         eval_set=[(x_train, y_train), (x_test, y_test)],        # eval_set=[(훈련set, 검증set)] : 훈련되는거 보여줌    # train set 명시해야 validation 지정 가능
         eval_metric='rmse',# 'mae', 'logloss']
)


# 4. 평가, 예측
results = model.score(x_test, y_test)
ic(results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
ic(r2)

print("=================================================================")
hist = model.evals_result()
ic(hist)    # XGB에서 제공



# eval_results의 그래프를 그려라.
import matplotlib.pyplot as plt
import numpy as np

results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')

ax.legend()
plt.ylabel('r2')
plt.title('XGBoost r2')
plt.show()

'''
ic| results: 0.9220259407074536
ic| r2: 0.9220259407074536
'''