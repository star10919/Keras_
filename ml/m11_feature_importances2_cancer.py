# feature = column = 열

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier     # RandomForestClassifier는 DecisionTreeClassifier의 앙상블 모델
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

### feature_importances
# 분류

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
# model = DecisionTreeClassifier(max_depth=5)
# model = RandomForestClassifier()
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
acc : 0.8947368421052632
[0.         0.06054151 0.         0.         0.         0.
 0.00636533 0.02005078 0.         0.         0.01257413 0.
 0.         0.00716099 0.         0.         0.02291518 0.00442037
 0.004774   0.         0.         0.01642816 0.         0.72839202
 0.         0.         0.         0.11637753 0.         0.        ]


* RandomForestClassifier
acc : 0.9649122807017544
[0.04594489 0.01396862 0.03249729 0.01881504 0.00589572 0.01088367
 0.0474708  0.12498551 0.00281653 0.00315305 0.01074243 0.00392675
 0.02002314 0.0399178  0.00331371 0.00354595 0.00358579 0.00318945
 0.0027572  0.00562694 0.17345586 0.02005404 0.15872429 0.06739186
 0.01508678 0.01327798 0.02374231 0.11083588 0.00650967 0.00786106]
'''