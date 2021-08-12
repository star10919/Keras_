# feature = column = 열

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier     # RandomForestClassifier는 DecisionTreeClassifier의 앙상블 모델
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

### feature_importances
### xgboost
# 분류

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
# model = DecisionTreeClassifier(max_depth=5)
model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc :', acc)
# acc : 0.9333333333333333

print(model.feature_importances_)       # feature_importances(컬럼의 중요도)_ : tree 계열에서 제공되는 강력한 애
# [0.0125026  0.         0.03213177 0.95536562] :  두번째 컬럼을 삭제해도 acc는 0.9333333333333333이 나옴



# 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()


'''
* DecisionTreeClassifier
acc : 0.9666666666666667
[0.         0.0125026  0.03213177 0.95536562]


* RandomForestClassifier
acc : 0.9333333333333333
[0.09735493 0.02836528 0.3902019  0.48407789]
'''