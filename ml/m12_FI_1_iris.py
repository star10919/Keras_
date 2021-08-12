# 실습
# 피처임포턴스가 전체 중요도 하위 20 ~ 25%인 컬럼들을 제거하여 데이터셋을 재구성한 후
# 각 모델별로 돌려서 결과 도출
# 기존 모델결과와 비교

# feature = column = 열

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier     # RandomForestClassifier는 DecisionTreeClassifier의 앙상블 모델
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from icecream import ic

### feature_importances
### xgboost
# 분류

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target



x = pd.DataFrame(x)

ic(x.shape, y.shape)

# Feature Importances 낮은 컬럼 삭제
x = x.drop([0], axis=1)
ic(x.shape)

x = x.to_numpy()



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)



# 2. 모델
model = DecisionTreeClassifier(max_depth=5)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc :', acc)
# acc : 0.9333333333333333

print(model.feature_importances_)       # feature_importances(컬럼의 중요도)_ : tree 계열에서 제공되는 강력한 애
# [0.0125026  0.         0.03213177 0.95536562] :  두번째 컬럼을 삭제해도 acc는 0.9333333333333333이 나옴


'''
<결과비교>
* DecisionTreeClassifier
삭제전
acc : 0.9666666666666667
[0.         0.0125026  0.03213177 0.95536562]
삭제후
acc : 0.9666666666666667
[0.0125026  0.53835801 0.44913938]


* RandomForestClassifier
삭제전
acc : 0.9333333333333333
[0.09735493 0.02836528 0.3902019  0.48407789]
삭제후
acc : 0.8666666666666667
[0.22568719 0.35740962 0.4169032 ]


* GradientBoostingClassifier
삭제전
acc : 0.9666666666666667
[0.00170799 0.01412577 0.26603707 0.71812917]
삭제후
acc : 0.9333333333333333
[0.01495726 0.22438981 0.76065293]


* XGBClassifier
삭제전
acc : 0.9
[0.01835513 0.0256969  0.62045246 0.3354955 ]
삭제후
acc : 0.9
[0.02876593 0.63379896 0.3374351 ]
'''