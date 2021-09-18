# 실습
# 피처임포턴스가 전체 중요도 하위 20 ~ 25%인 컬럼들을 제거하여 데이터셋을 재구성한 후
# 각 모델별로 돌려서 결과 도출
# 기존 모델결과와 비교
# feature = column = 열

from os import P_DETACH
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier     # RandomForestClassifier는 DecisionTreeClassifier의 앙상블 모델
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
from icecream import ic

### feature_importances
# 분류

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target




x = pd.DataFrame(x)

ic(x.shape, y.shape)
'''
x.shape: (178, 13), y.shape: (178,)
'''

# Feature Importances 낮은 컬럼 삭제
x = x.drop([5,8], axis=1)
ic(x.shape)

x = x.to_numpy()




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
# model = DecisionTreeClassifier(max_depth=5)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc :', acc)

print(model.feature_importances_)       # feature_importances(컬럼의 중요도)_ : tree 계열에서 제공되는 강력한 애

'''
* DecisionTreeClassifier
삭제전
acc : 0.9444444444444444
[0.         0.00489447 0.         0.0555874  0.         0.
 0.18739896 0.         0.         0.05677108 0.         0.33215293
 0.36319516]
삭제후
acc : 0.9444444444444444
[0.03534892 0.0555874  0.1569445  0.05677108 0.33215293 0.36319516]


* RandomForestClassifier
삭제전
acc : 1.0
[0.10102745 0.0327832  0.00970168 0.03889062 0.02928882 0.05304331
 0.1694222  0.0065932  0.02198911 0.14308796 0.0883063  0.12792962
 0.17793651]
삭제후
acc : 1.0
[0.10098497 0.02275325 0.02874257 0.0230868  0.03543703 0.15170541
 0.01985934 0.16207939 0.0804426  0.15502035 0.21988828]


 * GradientBoostingClassifier
 삭제전
acc : 0.9722222222222222
[1.76920491e-02 3.83073917e-02 1.99149330e-02 1.66141012e-02
 8.44639504e-04 2.44060507e-05 1.05354665e-01 6.76145641e-04
 2.13452691e-04 2.46481399e-01 2.84702712e-02 2.51919885e-01
 2.73486661e-01]
 삭제후
acc : 0.9722222222222222
[0.01738061 0.03954699 0.02046933 0.00754683 0.0023388  0.10588875
 0.00066469 0.2495337  0.02916213 0.25202789 0.27544027]


 * XGBClassifier
 삭제전
acc : 1.0
[0.01854127 0.04139536 0.01352911 0.01686821 0.02422602 0.00758254
 0.10707161 0.01631111 0.00051476 0.12775211 0.01918284 0.50344414
 0.10358089]
 삭제후
acc : 1.0
[0.02633043 0.04068652 0.01340542 0.02495279 0.02954447 0.1471631
 0.01496556 0.14296433 0.02313021 0.39706442 0.13979274]
'''