### 데이터 증폭(smote 사용) - acc보다 F1 score가 높아짐
# y 라벨 개수 가장 큰 거 기준으로 동일하게 맞춰줌

from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (178, 13) (178,)

print(pd.Series(y).value_counts())      #value_counts는 판다스 함수임!(넘파이X)
# 1    71
# 0    59
# 2    48

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_new = x[:-30]
y_new = y[:-30]
print(x_new.shape, y_new.shape)     #  (178, 13) (178,) -> (148, 13) (148,)
print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, train_size=0.75, shuffle=True, random_state=9, stratify=y_new)    # stratify=y_new : y_new 라벨의 비율로 나눠줌!!!
print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2    14

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')     # xgboost 쓰면 이발메트릭스 사용해야 함!

score = model.score(x_test, y_test)
print("model.score :", score)       # model.score : 0.9459459459459459


########################################### smote 적용 ##############################################
print("=============================== smote 적용 ===============================")

smote = SMOTE(random_state=66)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)     # train만 smote(증폭) 시킴, test는 하지 않음
#####################################################################################################

print(pd.Series(y_smote_train).value_counts())
# 0    53
# 1    53
# 2    53
print(x_smote_train.shape, y_smote_train.shape)     # (159, 13) (159,)

print("smote 전 :", x_train.shape, y_train.shape)
print("smote 후 :", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 :\n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 :\n", pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print("model2.score :", score2)      # model2.score : 0.972972972972973