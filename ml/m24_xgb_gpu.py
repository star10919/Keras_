from sklearn import datasets
from  xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from icecream import ic

### CPU(머신러닝 디폴트), GPU 선택할 수 있음
#  GPU : (#2.모델에서) tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor'


# 1. 데이
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
model = XGBRegressor(n_estimators=10000, learing_rate=0.01, n_jobs=-1,# n_jobs = 코어 수
                    tree_method='gpu_hist', gpu_id=0,
                    predictor='gpu_predictor'   #cpu_predictor
                    )        




# 3. 훈련
import time
start = time.time()
model.fit(x_train, y_train, verbose=1,                           # verbose=1 : eval_set 보여줌
         eval_set=[(x_train, y_train), (x_test, y_test)],        # eval_set=[(훈련set, 검증set)] : 훈련되는거 보여줌    # train set 명시해야 validation 지정 가능
         eval_metric='rmse',# 'mae', 'logloss']

)
end = time.time() - start
print("걸린시간 :", end)

# 17-9700 / 2080ti
# n_jobs=1 걸린시간 : 4.701421499252319
# n_jobs=2 걸린시간 : 4.202755928039551
# n_jobs=5 걸린시간 : 4.502980947494507
# n_jobs=8 걸린시간 : 4.310497999191284
# n_jobs=-1 걸린시간 : 4.365349292755127