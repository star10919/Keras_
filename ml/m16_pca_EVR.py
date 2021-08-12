import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from icecream import ic

### pca.explained_variance_ratio_ : 피쳐임포턴스가 낮은순서대로 압축됨. 피쳐임포턴스 높은순서대로 보여줌

# 1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)       # (442, 10) (442,)

pca = PCA(n_components=10)   # 컬럼을 7개로 압축(삭제 아님)하겠다! (원래 컬럼(10)보다 많으면 에러남)
x = pca.fit_transform(x)
ic(x)
ic(x.shape)                # (442, 7)

pca_EVR = pca.explained_variance_ratio_     # 피쳐임포턴스가 낮은순서대로 압축됨. 피쳐임포턴스 높은순서대로 보여줌
ic(pca_EVR)
ic(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)     # 피쳐 임포턴스 누적해서 보여줌
print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]

print(np.argmax(cumsum >= 0.94)+1)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()



'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66, shuffle=True)


# 2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 :", results)



* XGB 결과 : 0.28428734040184866
* PCA 결과 : 0.3538690321164464
'''