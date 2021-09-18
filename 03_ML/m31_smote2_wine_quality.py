### 데이터 증폭(smote 사용) - acc보다 F1 score가 높아짐
# y 라벨 개수 가장 큰 거 기준으로 동일하게 맞춰줌
### k_neighbors 의 디폴트 5 : 각 라벨의 개수가 5보다 커야 증폭(smote) 가능해짐! / 아니면 k_neighbors의 값을 낮춰주면 됨
#   => k_neighbors 값 줄이면 score 떨어짐(연산수 줄기 때문에)


from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=9, stratify=y)    # stratify=y_new : y_new 라벨의 비율로 나눠줌!!!
print(pd.Series(y_train).value_counts())
#  6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')     # xgboost 쓰면 이발메트릭스 사용해야 함!

score = model.score(x_test, y_test)
print("model.score :", score)       # model.score : 0.6563265306122449


########################################### smote 적용 ##############################################
print("=============================== smote 적용 ===============================")
smote = SMOTE(random_state=66, k_neighbors=3)       #k_neighbors 의 디폴트 5  / 가장 작은 라벨인 9의 라벨 개수가 4이므로 디폴드인 5보다 작은 값으로 에러나니까   k_neighbors의 값을 낮춰줌

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)     # train만 smote(증폭) 시킴, test는 하지 않음
#####################################################################################################

print(pd.Series(y_smote_train).value_counts())
# 6.0    1648
# 5.0    1648
# 4.0    1648
# 9.0    1648
# 8.0    1648
# 7.0    1648
# 3.0    1648
print(x_smote_train.shape, y_smote_train.shape)     # 

print("smote 전 :", x_train.shape, y_train.shape)
print("smote 후 :", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 :\n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 :\n", pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print("model2.score :", score2)      # model2.score : 0.6302040816326531