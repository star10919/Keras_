# 실습
# 피처임포턴스가 전체 중요도 하위 20 ~ 25%인 컬럼들을 제거하여 데이터셋을 재구성한 후
# 각 모델별로 돌려서 결과 도출
# 기존 모델결과와 비교
# feature = column = 열

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from icecream import ic

### feature_importances
# 회귀

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target




x = pd.DataFrame(x)

ic(x.shape, y.shape)
'''
 x.shape: (506, 13), y.shape: (506,)
'''

# Feature Importances 낮은 컬럼 삭제
x = x.drop([1,3], axis=1)
ic(x.shape)

x = x.to_numpy()





x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
# model = DecisionTreeRegressor(max_depth=5)
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc :', acc)


print(model.feature_importances_)       # feature_importances(컬럼의 중요도)_ : tree 계열에서 제공되는 강력한 애



'''
* DecisionTreeClassifier
삭제전
acc : 0.8507309980875365
[0.04699249 0.         0.         0.         0.01464736 0.29899984
 0.         0.05933307 0.         0.00583002 0.         0.
 0.57419722]
삭제후
acc : 0.7539226674500856
[0.04639969 0.29160919 0.0840373  0.00813405 0.56981977]


* RandomForestClassifier
삭제전
acc : 0.9238697367819444
[0.04041184 0.00109653 0.00612171 0.00083303 0.02584168 0.41724409
 0.01150053 0.0656355  0.00445457 0.01447544 0.01770787 0.0119425
 0.38273472]
삭제후
acc : 0.9166545383258504
[0.0460146  0.023749   0.38655075 0.01601079 0.07399649 0.02148195
 0.01303366 0.41916277]


 * GradientBoostingRegressor
 삭제전
 acc : 0.9455879420898174
[2.43689159e-02 2.13083076e-04 4.98615415e-03 1.78017906e-04
 4.11408446e-02 3.58304271e-01 5.95477444e-03 8.41256669e-02
 2.34942091e-03 1.10710640e-02 3.11529030e-02 6.73480598e-03
 4.29420078e-01]
 삭제후
acc : 0.9458372360047498
[0.02407291 0.00605501 0.04116857 0.35757565 0.00653343 0.08307824
 0.00244057 0.01114388 0.03115497 0.00677311 0.43000367]


 * XGBRegressor
 삭제전
 acc : 0.9221188601856797
[0.01447935 0.00363372 0.01479119 0.00134153 0.06949984 0.30128643
 0.01220458 0.0518254  0.0175432  0.03041655 0.04246345 0.01203115
 0.42848358]
 삭제후
acc : 0.9203131474915633
[0.01447387 0.01181944 0.0553927  0.29302305 0.01343597 0.05159502
 0.01615641 0.03287613 0.04188036 0.01217921 0.4571678 ]

'''