# 실습
# cancer로 만들 것
# 지표는 f1
# 라벨 0을 112개 삭제

from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (569, 30) (569,)

# print(y)

y = y.reshape(569,1)
print(y.shape)      # (569, 1)

# x, y 결합(정렬하기 위해서)
xy_concat = np.concatenate((x,y), axis=1)
print(xy_concat.shape)      # (569, 31)

# y값 기준 정렬(0->1)
xy_concat = xy_concat[xy_concat[:, 30].argsort()]
print(xy_concat)

x = xy_concat[112:, :-1] 
y = xy_concat[112:, -1]

print(x.shape, y.shape)     # (457, 30) (457,)
print(y)

print(pd.Series(y).value_counts())      #value_counts는 판다스 함수임!(넘파이X)
# 1.0    357
# 0.0    100

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=66)#, stratify=y)    # stratify=y_new : y_new 라벨의 비율로 나눠줌!!!
print(pd.Series(y_train).value_counts())
# 1.0    272
# 0.0     70

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')     # xgboost 쓰면 이발메트릭스 사용해야 함!

score = model.score(x_test, y_test)
print("model.score :", score)       # model.score : 0.9304347826086956

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("f1_score :", f1)     # f1_score : 0.9550561797752808


########################################### smote 적용 ##############################################
print("=============================== smote 적용 ===============================")
smote = SMOTE(random_state=66)#, k_neighbors=61)       #k_neighbors 의 디폴트 5  / 가장 작은 라벨인 9의 라벨 개수가 4이므로 디폴드인 5보다 작은 값으로 에러나니까   k_neighbors의 값을 낮춰줌

start = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)     # train만 smote(증폭) 시킴, test는 하지 않음
end = time.time() - start
#####################################################################################################

print(pd.Series(y_smote_train).value_counts())
# 0.0    272
# 1.0    272
print(x_smote_train.shape, y_smote_train.shape)     # (544, 30) (544,)

print("smote 전 :", x_train.shape, y_train.shape)
print("smote 후 :", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 :\n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 :\n", pd.Series(y_smote_train).value_counts())
print("SMOTE 경과시간 :", end)

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print("model2.score :", score2)      # model2.score : 0.9478260869565217

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("f1_score :", f1)     # f1_score : 0.9659090909090909
