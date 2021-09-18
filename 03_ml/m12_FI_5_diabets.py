# 실습
# 피처임포턴스가 전체 중요도 하위 20 ~ 25%인 컬럼들을 제거하여 데이터셋을 재구성한 후
# 각 모델별로 돌려서 결과 도출
# 기존 모델결과와 비교
# feature = column = 열

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
from icecream import ic

### feature_importances
# 회귀

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target



x = pd.DataFrame(x)

ic(x.shape, y.shape)
'''
x.shape: (442, 10), y.shape: (442,)
'''

# Feature Importances 낮은 컬럼 삭제
x = x.drop([0,7], axis=1)
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
acc : 0.18699053453135217
[0.04339214 0.         0.24919201 0.11505227 0.         0.04366568
 0.03928846 0.         0.45566222 0.05374722]
삭제후
acc : 0.18699053453135217
[0.04339214 0.24919201 0.11505227 0.04366568 0.0403601  0.45459058
 0.05374722]


* RandomForestClassifier
삭제전
acc : 0.3559954566609884
[0.07067494 0.01128309 0.26853167 0.10536462 0.04283663 0.05354345
 0.04516689 0.02149173 0.309696   0.07141098]
삭제후
acc : 0.3619488035467898
[0.07152647 0.28476969 0.10380682 0.0446664  0.06051003 0.05552172
 0.30363705 0.07556181]


* GradientBoostingRegressor
삭제전
acc : 0.3883629740643214
[0.05958658 0.01125969 0.27493593 0.11803521 0.02499367 0.05345065
 0.03875504 0.01491835 0.34355126 0.06051361]
삭제후
acc : 0.3305461363082919
[0.06081073 0.2776071  0.12094716 0.07868871 0.03967426 0.35554358
 0.06672847]


* XGBRegressor
삭제전
acc : 0.23802704693460175
[0.02593722 0.03821947 0.19681752 0.06321313 0.04788675 0.05547737
 0.07382318 0.03284872 0.3997987  0.06597802]
삭제후
acc : 0.24227362402135721
[0.0375583  0.2132655  0.08872849 0.05603264 0.10501637 0.07736687
 0.35352597 0.06850582]

'''