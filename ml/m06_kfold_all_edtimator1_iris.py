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
from sklearn.model_selection import KFold, cross_val_score


# 2. 모델(머신러닝에서는 정의만 해주면 됨)
# from sklearn.utils.testing import all_estimators
from sklearn.utils import all_estimators        # 이걸로 바뀜(testing 빠짐)
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')               ### warning 무시

allAlgorithms = all_estimators(type_filter='classifier')      # 분류모델
# allAlgorithms = all_estimators(type_filter='regressor')         # 회귀모델
# ic(allAlgorithms)
print('모델의 개수 :', len(allAlgorithms))      # 모델의 개수 : 41

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for ( name, algorithm ) in allAlgorithms:
    try :
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, '평균 :', round(np.mean(scores), 4))

    except:         
        # continue      # 에러뜨는거 제외하고 정상적인 거만 돌아감
        print(name, "은 없는 놈!!!")

'''
모델의 개수 : 41
AdaBoostClassifier [0.63333333 0.93333333 1.         0.9        0.96666667] 평균 : 0.8867
BaggingClassifier [0.93333333 0.93333333 1.         0.9        0.96666667] 평균 : 0.9467
BernoulliNB [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 : 0.2933
CalibratedClassifierCV [0.9        0.83333333 1.         0.86666667 0.96666667] 평균 : 0.9133
CategoricalNB [0.9        0.93333333 0.93333333 0.9        1.        ] 평균 : 0.9333
ClassifierChain 은 없는 놈!!!
ComplementNB [0.66666667 0.66666667 0.7        0.6        0.7       ] 평균 : 0.6667
DecisionTreeClassifier [0.96666667 0.96666667 1.         0.9        0.93333333] 평균 : 0.9533
DummyClassifier [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 : 0.2933
ExtraTreeClassifier [0.83333333 0.93333333 0.96666667 0.86666667 0.93333333] 평균 : 0.9067
ExtraTreesClassifier [0.96666667 0.96666667 1.         0.86666667 0.96666667] 평균 : 0.9533
GaussianNB [0.96666667 0.9        1.         0.9        0.96666667] 평균 : 0.9467
GaussianProcessClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 : 0.96
GradientBoostingClassifier [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 : 0.9667
HistGradientBoostingClassifier [0.86666667 0.96666667 1.         0.9        0.96666667] 평균 : 0.94
KNeighborsClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 : 0.96
LabelPropagation [0.93333333 1.         1.         0.9        0.96666667] 평균 : 0.96
LabelSpreading [0.93333333 1.         1.         0.9        0.96666667] 평균 : 0.96
LinearDiscriminantAnalysis [1.  1.  1.  0.9 1. ] 평균 : 0.98
LinearSVC [0.96666667 0.96666667 1.         0.9        1.        ] 평균 : 0.9667
LogisticRegression [1.         0.96666667 1.         0.9        0.96666667] 평균 : 0.9667
LogisticRegressionCV [1.         0.96666667 1.         0.9        1.        ] 평균 : 0.9733
MLPClassifier [1.         0.96666667 1.         0.93333333 1.        ] 평균 : 0.98
MultiOutputClassifier 은 없는 놈!!!
MultinomialNB [0.96666667 0.93333333 1.         0.93333333 1.        ] 평균 : 0.9667
NearestCentroid [0.93333333 0.9        0.96666667 0.9        0.96666667] 평균 : 0.9333
NuSVC [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 : 0.9733
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier [0.96666667 0.86666667 1.         0.93333333 1.        ] 평균 : 0.9533
Perceptron [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 평균 : 0.78
QuadraticDiscriminantAnalysis [1.         0.96666667 1.         0.93333333 1.        ] 평균 : 0.98
RadiusNeighborsClassifier [0.96666667 0.9        0.96666667 0.93333333 1.        ] 평균 : 0.9533
RandomForestClassifier [0.93333333 0.96666667 1.         0.9        0.96666667] 평균 : 0.9533
RidgeClassifier [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 : 0.84
RidgeClassifierCV [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 : 0.84
SGDClassifier [0.73333333 0.83333333 0.96666667 0.9        1.        ] 평균 : 0.8867
SVC [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 : 0.9667
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
'''