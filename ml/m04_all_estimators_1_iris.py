import warnings
import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### all_estimators(모든 알고리즘(모델) 사용)

datasets = load_iris()

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# 1-2. 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# 2. 모델(머신러닝에서는 정의만 해주면 됨)
# from sklearn.utils.testing import all_estimators
from sklearn.utils import all_estimators        # 이걸로 바뀜(testing 빠짐)
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')               ### warning 무시

allAlgorithms = all_estimators(type_filter='classifier')      # 분류모델
# allAlgorithms = all_estimators(type_filter='regressor')         # 회귀모델
# ic(allAlgorithms)
print('모델의 개수 :', len(allAlgorithms))      # 모델의 개수 : 41


for ( name, algorithm ) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)
        # print(name)

        y_predict = model.predict(x_test)   #score 안먹히는 거 있어서    predict 사용
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 :', acc)
    
    except:         
        # continue      # 에러뜨는거 제외하고 정상적인 거만 돌아감
        print(name, "은 없는 놈!!!")

'''
모델의 개수 : 41
AdaBoostClassifier 의 정답률 : 0.9111111111111111
BaggingClassifier 의 정답률 : 0.8888888888888888
BernoulliNB 의 정답률 : 0.7555555555555555
CalibratedClassifierCV 의 정답률 : 0.8222222222222222
CategoricalNB 은 없는 놈!!!
ClassifierChain 은 없는 놈!!!
ComplementNB 은 없는 놈!!!
DecisionTreeClassifier 의 정답률 : 0.9111111111111111
DummyClassifier 의 정답률 : 0.28888888888888886
ExtraTreeClassifier 의 정답률 : 0.9555555555555556
ExtraTreesClassifier 의 정답률 : 0.9555555555555556
GaussianNB 의 정답률 : 0.9555555555555556
GaussianProcessClassifier 의 정답률 : 0.9777777777777777
GradientBoostingClassifier 의 정답률 : 0.8888888888888888
HistGradientBoostingClassifier 의 정답률 : 0.9111111111111111
KNeighborsClassifier 의 정답률 : 0.8888888888888888
LabelPropagation 의 정답률 : 0.9111111111111111
LabelSpreading 의 정답률 : 0.9111111111111111
LinearDiscriminantAnalysis 의 정답률 : 1.0
LinearSVC 의 정답률 : 0.9111111111111111
LogisticRegression 의 정답률 : 0.9777777777777777
LogisticRegressionCV 의 정답률 : 1.0
MLPClassifier 의 정답률 : 0.9555555555555556
MultiOutputClassifier 은 없는 놈!!!
MultinomialNB 은 없는 놈!!!
NearestCentroid 의 정답률 : 0.8666666666666667
NuSVC 의 정답률 : 0.9555555555555556
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.8
Perceptron 의 정답률 : 0.8222222222222222
QuadraticDiscriminantAnalysis 의 정답률 : 1.0
RadiusNeighborsClassifier 은 없는 놈!!!
RandomForestClassifier 의 정답률 : 0.8888888888888888
RidgeClassifier 의 정답률 : 0.8222222222222222
RidgeClassifierCV 의 정답률 : 0.8222222222222222
SGDClassifier 의 정답률 : 1.0
SVC 의 정답률 : 0.9555555555555556
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
'''