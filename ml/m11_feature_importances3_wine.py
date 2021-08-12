# feature = column = 열

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier     # RandomForestClassifier는 DecisionTreeClassifier의 앙상블 모델
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

### feature_importances
# 분류

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
# model = DecisionTreeClassifier(max_depth=5)
model = RandomForestClassifier()
model = GradientBoostingClassifier()
# model = XGBClassifier()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc :', acc)

print(model.feature_importances_)       # feature_importances(컬럼의 중요도)_ : tree 계열에서 제공되는 강력한 애


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
acc : 0.9444444444444444
[0.         0.00489447 0.         0.0555874  0.         0.
 0.18739896 0.         0.         0.05677108 0.         0.33215293
 0.36319516]


* RandomForestClassifier
acc : 1.0
[0.10102745 0.0327832  0.00970168 0.03889062 0.02928882 0.05304331
 0.1694222  0.0065932  0.02198911 0.14308796 0.0883063  0.12792962
 0.17793651]
'''