from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel

### 과적합 줄이는 법 - (SelectFromModel사용)안 좋은 feature을 솎아낸다!!


# 1. 데이터
# datasets = load_boston()
# x = datasets.data
# y = datasets.target
x, y = load_boston(return_X_y=True) #return_X_y=True  : x, y 분류돼서 나옴
print(x.shape, y.shape)     #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True
)


# 2. 모델
model = XGBRegressor(n_jobs=8)


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score = model.score(x_test, y_test)
print("model.score :", score)           #model.score : 0.9221188601856797

threshold = np.sort(model.feature_importances_)
print(threshold)
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

print("======================================")
for thresh in threshold:
    # print(thres)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)   # thresh에 컬럼 하나씩 들어가는데, 그 컬럼 피쳐임포턴스보다 이상인 컬럼들만 추출함
    # print(selection)

    # threshold=thresh 이상의 컬럼들로 train,test데이터 재구성(SelectFromModel:컬럼삭제해줌!)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thres=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))

'''
(506, 13) (506,)
model.score : 0.9221188601856797
[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]
======================================
(404, 13) (102, 13)
Thres=0.001, n=13, R2: 92.21%       전체컬럼
(404, 12) (102, 12)
Thres=0.004, n=12, R2: 92.16%       컬럼1개삭제
(404, 11) (102, 11)
Thres=0.012, n=11, R2: 92.03%       컬럼2개삭제
(404, 10) (102, 10)
Thres=0.012, n=10, R2: 92.19%       컬럼3개삭제
(404, 9) (102, 9)
Thres=0.014, n=9, R2: 93.08%    *** 컬럼4개삭제
(404, 8) (102, 8)
Thres=0.015, n=8, R2: 92.37%        컬럼5개삭제
(404, 7) (102, 7)
Thres=0.018, n=7, R2: 91.48%        컬럼6개삭제
(404, 6) (102, 6)
Thres=0.030, n=6, R2: 92.71%        컬럼7개삭제
(404, 5) (102, 5)
Thres=0.042, n=5, R2: 91.74%        컬럼8개삭제
(404, 4) (102, 4)
Thres=0.052, n=4, R2: 92.11%        컬럼9개삭제
(404, 3) (102, 3)
Thres=0.069, n=3, R2: 92.52%        컬럼10개삭제
(404, 2) (102, 2)
Thres=0.301, n=2, R2: 69.41%        컬럼11개삭제
(404, 1) (102, 1)
Thres=0.428, n=1, R2: 44.98%        컬럼12개삭제
'''