# 실습, 모델구성하고 완료하시오.
# 회귀 데이터를 Classigier로 만들었을 경우에 에러 확인!!!

# 클래시파이가 아니므로 에러가 날거임

from sklearn.svm import LinearSVC, SVC      # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor     # 분류면 KNeighborsClassifier, 회귀면 KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression     # *** LogisticRegression : 분류모델 임!!!!!!!!!!!!!!!!!(이름에 Regression이 들어간다고 회귀모델 아님!!!!!!!!!!!!!!)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # 의사결정나무
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor