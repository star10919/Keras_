# 실습
# cancer로 만들 것
# 지표는 f1

from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (569, 30) (569,)

print(pd.Series(y).value_counts())      #value_counts는 판다스 함수임!(넘파이X)
# 1    357
# 0    212

print(y)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=66)#, stratify=y)    # stratify=y_new : y_new 라벨의 비율로 나눠줌!!!
print(pd.Series(y_train).value_counts())
# 1    265
# 0    161

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')     # xgboost 쓰면 이발메트릭스 사용해야 함!

score = model.score(x_test, y_test)
print("model.score :", score)       # model.score : 0.9790209790209791

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("f1_score :", f1)     # f1_score : 0.9837837837837837


########################################### smote 적용 ##############################################
print("=============================== smote 적용 ===============================")
smote = SMOTE(random_state=66)#, k_neighbors=61)       #k_neighbors 의 디폴트 5  / 가장 작은 라벨인 9의 라벨 개수가 4이므로 디폴드인 5보다 작은 값으로 에러나니까   k_neighbors의 값을 낮춰줌

start = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)     # train만 smote(증폭) 시킴, test는 하지 않음
end = time.time() - start
#####################################################################################################

print(pd.Series(y_smote_train).value_counts())
# 0    265
# 1    265
print(x_smote_train.shape, y_smote_train.shape)     # (530, 30) (530,)

print("smote 전 :", x_train.shape, y_train.shape)
print("smote 후 :", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 :\n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 :\n", pd.Series(y_smote_train).value_counts())
print("SMOTE 경과시간 :", end)

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print("model2.score :", score2)      # model2.score : 0.986013986013986

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("f1_score :", f1)     # f1_score : 0.989247311827957
