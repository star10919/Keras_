# feature = column = 열

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

### feature_importances
# 회귀

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델
# model = DecisionTreeRegressor(max_depth=5)
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc :', acc)


print(model.feature_importances_)       # feature_importances(컬럼의 중요도)_ : tree 계열에서 제공되는 강력한 애


'''
* DecisionTreeClassifier
acc : 0.18699053453135217
[0.04339214 0.         0.24919201 0.11505227 0.         0.04366568
 0.03928846 0.         0.45566222 0.05374722]


* RandomForestClassifier
acc : 0.3559954566609884
[0.07067494 0.01128309 0.26853167 0.10536462 0.04283663 0.05354345
 0.04516689 0.02149173 0.309696   0.07141098]
'''