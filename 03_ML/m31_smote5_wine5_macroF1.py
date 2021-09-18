from imblearn.over_sampling import SMOTE
from numpy.lib.function_base import average
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

### macro_F1 : f1의 평균
# 3, 4 -> 0
# 5, 6, 7 -> 1
# 8, 9 -> 2

# 변경
# 3, 4, 5 -> 0
# 6 -> 1
# 7, 8, 9 -> 2

datasets = pd.read_csv('../_data/winequality-white.csv', index_col=None, header=0, sep=';')     # 비정제 데이터

datasets = datasets.values      # 판다스 넘파이로 변환
x = datasets[:, :11]
y = datasets[:, 11]
print(x.shape, y.shape)     # (4898, 11) (4898,)

print(pd.Series(y).value_counts())      #value_counts는 판다스 함수임!(넘파이X)
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5

print(y)

####################################################################
##### 라벨 대통합!! 라벨 9의 값이 5가 되지 않기 때문에 8에 통합
####################################################################
print('================= 라벨 통합 =================')
y_new = []
for index, value in enumerate (y):
    if value == 9:
        y[index] = 2
    elif value == 8:
        y[index] = 2
    elif value == 7:
        y[index] = 2
    elif value == 6:
        y[index] = 1
    elif value == 5:
        y[index] = 0
    elif value == 4:
        y[index] = 0
    elif value == 3:
        y[index] = 0



print(pd.Series(y).value_counts())
'''
1.0    2198
0.0    1640
2.0    1060
'''




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=66)#, stratify=y)    # stratify=y_new : y_new 라벨의 비율로 나눠줌!!!
print(pd.Series(y_train).value_counts())
# 1.0    1641
# 0.0    1253
# 2.0     779

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')     # xgboost 쓰면 이발메트릭스 사용해야 함!

score = model.score(x_test, y_test)
print("model.score :", score)       # model.score : 0.7061224489795919

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score :", f1)     # f1_score : 0.7051495862438939


########################################### smote 적용 ##############################################
print("=============================== smote 적용 ===============================")
smote = SMOTE(random_state=66)#, k_neighbors=61)       #k_neighbors 의 디폴트 5  / 가장 작은 라벨인 9의 라벨 개수가 4이므로 디폴드인 5보다 작은 값으로 에러나니까   k_neighbors의 값을 낮춰줌

start = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)     # train만 smote(증폭) 시킴, test는 하지 않음
end = time.time() - start
#####################################################################################################

print(pd.Series(y_smote_train).value_counts())
# 0.0    1641
# 1.0    1641
# 2.0    1641
print(x_smote_train.shape, y_smote_train.shape)     # (4923, 11) (4923,)

print("smote 전 :", x_train.shape, y_train.shape)
print("smote 후 :", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 :\n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 :\n", pd.Series(y_smote_train).value_counts())
print("SMOTE 경과시간 :", end)

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print("model2.score :", score2)      # model2.score : 0.6995918367346938

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score :", f1)     # f1_score : 0.7009036847917063
