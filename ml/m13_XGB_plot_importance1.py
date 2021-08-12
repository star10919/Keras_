# feature = column = 열

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost.plotting import plot_importance


### feature_importances
# 회귀

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
# model = DecisionTreeRegressor(max_depth=5)
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc :', acc)


print(model.feature_importances_)       # feature_importances(컬럼의 중요도)_ : tree 계열에서 제공되는 강력한 애


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
* DecisionTreeClassifier
acc : 0.8507309980875365
[0.04699249 0.         0.         0.         0.01464736 0.29899984
 0.         0.05933307 0.         0.00583002 0.         0.
 0.57419722]


* RandomForestClassifier
acc : 0.9238697367819444
[0.04041184 0.00109653 0.00612171 0.00083303 0.02584168 0.41724409
 0.01150053 0.0656355  0.00445457 0.01447544 0.01770787 0.0119425
 0.38273472]
'''