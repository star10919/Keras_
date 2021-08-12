import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

### pca : 주성분 분석,, 차원축소(컬럼축소), 고차원의 데이터를 저차원으로 환원, 데이터전처리 중 하나임

# 1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)       # (442, 10) (442,)

pca = PCA(n_components=9)   # 컬럼을 7개로 압축(삭제 아님)하겠다! (원래 컬럼(10)보다 많으면 에러남)
x = pca.fit_transform(x)
print(x)
print(x.shape)              # (442, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66, shuffle=True)


# 2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 :", results)


'''
* XGB 결과 : 0.28428734040184866
* PCA 결과 : 0.3538690321164464
'''