from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel

# 실습
# 1. 상단모델에 그리드서치, 랜덤서치로 튜닝한 모델 구성 최적의 R2 값과 피쳐임포턴스 구할 것

# 2. 위 스레드값으로 SelectFromModel 돌려서 최적의 피쳐 갯수 구할것

# 3. 위 피쳐 갯수로 피쳐 갯수를 조정한 뒤 그걸도 다시 랜덤서치 그리드서치해서 최적의 R2값 구할 것

# 4. 1번과 3번값 비교



### 과적합 줄이는 법 - (SelectFromModel사용)안 좋은 feature을 솎아낸다!!

# 1. 데이터
# datasets = load_boston()
# x = datasets.data
# y = datasets.target
x, y = load_diabetes(return_X_y=True) #return_X_y=True  : x, y 분류돼서 나옴
print(x.shape, y.shape)     #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

parameters = [
    {'n_estimators':[100, 150, 200], 'max_depth':[6, 8], 'min_samples_split':[3, 5]},
    {'max_depth':[6, 8, 12], 'min_samples_split':[2, 3, 5]},
    {'min_samples_leaf':[5], 'min_samples_split':[5, 10], 'max_depth':[6, 8, 10]},
    {'n_jobs':[-1, 2, 4], 'min_samples_split':[2, 3, 5, 10]}
]


# 2. 모델
# model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)    #best_score : 0.49495939028073366
model = RandomForestRegressor(max_depth=6, min_samples_split=3, n_estimators=200)

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
# print("최적의 매개변수 :", model.best_estimator_)
# print("최적의 매개변수 :", model.best_params_)
# print("best_score :", model.best_score_)

score = model.score(x_test, y_test)
print("model.score :", score)           #model.score : 0.39892452549058044



threshold = np.sort(model.feature_importances_)
print(threshold)


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


