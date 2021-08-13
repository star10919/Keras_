# 실습
# m13로 만든 0.95 이상의 n_component=? 를 사용하여 xgb 모델을 만들 것(default)
# mnist dnn 보다 성능 좋게 만들어라!!!
# DNN, CNN 과 비교!!!

# RandomSearch로도 해 볼 것

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost.plotting import plot_importance
from sklearn.decomposition import PCA
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


### feature_importances
# 분류

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

ic(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
ic(y_train.shape, y_test.shape)      # (60000,) (10000,)

x = np.append(x_train, x_test, axis=0)
ic(x.shape)      # x.shape: (70000, 28, 28)

y= np.append(y_train, y_test, axis=0)
ic(y.shape)      # y.shape: (70000,)
ic(np.unique(y))    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




x = x.reshape(x.shape[0], 28*28)
ic(x.shape)     # (70000, 784)

# PCA
pca = PCA(n_components=486)
x = pca.fit_transform(x)
ic(x)
ic(x.shape)                # (70000, 486)





# pca_EVR = pca.explained_variance_ratio_     # 피쳐임포턴스가 낮은순서대로 압축됨. 피쳐임포턴스 높은순서대로 보여줌
# ic(pca_EVR)
# ic(sum(pca_EVR))

# cumsum = np.cumsum(pca_EVR)     # 피쳐 임포턴스 낮은거부터 누적해서 보여줌
# ic(cumsum)

# ic(np.argmax(cumsum >= 0.999)+1)     # 486 # 전체 컬럼 수 n_components에 넣어서 원하는 피쳐임포턴스 수치의 컬럼 수 찾고, 나온 수 n_components에서 넣어주면 됨

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()






x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)


parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.0001, 0.01], "max_depth":[4, 5, 6]},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.5], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9], "n_jobs":[-1]}
]




# 2. 모델
# model = RandomizedSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1)
model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1)



# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
# print("최적의 매개변수 :", model.best_estimator_)
# print("최적의 매개변수 :", model.best_params_)
# print("best_score :", model.best_score_)

r2 = model.score(x_test, y_test)
print('r2 :', r2)


# print(model.feature_importances_)       # feature_importances(컬럼의 중요도)_ : tree 계열에서 제공되는 강력한 애


# 시각화
import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#              align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()

plot_importance(model)
plt.show()

'''
* RandomCV
최적의 매개변수 : {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1}
best_score : 0.8564905175957354
r2 : 0.850635316842861

* (최적의 매개변수 사용) XGBRegressor
r2 : 0.8530794615222785
'''