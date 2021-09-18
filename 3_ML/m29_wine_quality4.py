import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import Scalar
from sklearn.datasets import load_wine    # 정제된 데이터
from xgboost import XGBClassifier

### y 그룹화(7 label  ->  3 label)
### 3,4 = 0  /  5,6,7 = 1  /  8,9 =2

# 1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', index_col=None, header=0, sep=';')     # 비정제 데이터
print(datasets.head())
print(datasets.tail())
print(datasets.shape)     # (4898, 12)
print(datasets.describe())

datasets = datasets.values      # 넘파이로 전환
# print(type(datasets))       # <class 'numpy.ndarray'>
# print(datasets.shape)       # (4898, 12)

x = datasets[:, :11]
y = datasets[:, 11] 
print(y.shape)  # (4898,)

############################################################################
newlist = []
for i in list(y):
    if i<= 4:
        newlist +=[0]
    elif i<=7:
        newlist +=[1]
    else:
        newlist +=[2]
y = np.array(newlist)       # 리스트를 넘파이로 전환
print(y.shape)  # (4898,)
############################################################################

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)




import matplotlib.pyplot as plt
# datasets의 바 그래프를 그시시오!!!
# count_data = datasets.groupby('quality')['quality'].count()
# print(count_data)
# '''
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5
# '''
# # count_data.plot()
# plt.bar(count_data.index, count_data)   #(x, y)
# plt.show()



scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
model = XGBClassifier(n_jobs=-1)


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score = model.score(x_test, y_test)

print("accuracy :", score)


'''
y label 그룹화 전
accuracy : 0.6816326530612244

y label 그룹화 후
accuracy : 0.9469387755102041
'''