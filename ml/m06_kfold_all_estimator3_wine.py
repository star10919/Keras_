import warnings
import numpy as np
import pandas as pd
from icecream import ic
from pandas.core.tools.datetimes import Scalar
from tensorflow.python.keras.layers.recurrent import LSTM

### all_estimators(모든 알고리즘(모델) 사용)
# 다중분류


# 1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',       # 경로잡기 중요!
                        index_col=None, header=0)    #header=0 첫번째라인   # (4898,12)

datasets_np = datasets.to_numpy()   #1 판다스 -> 넘파이
x = datasets_np[:,0:11]
y = datasets_np[:,[-1]]


from sklearn.model_selection import KFold, cross_val_score


# 2. 모델 구성
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
AdaBoostClassifier [0.41428571 0.45       0.42244898 0.36261491 0.43615935] 평균 : 0.4171
BaggingClassifier [0.66632653 0.64183673 0.65918367 0.6659857  0.65270684] 평균 : 0.6572
BernoulliNB [0.45816327 0.43367347 0.44285714 0.46271706 0.44637385] 평균 : 0.4488
CalibratedClassifierCV [0.51326531 0.47142857 0.49693878 0.55158325 0.48416752] 평균 : 0.5035
CategoricalNB [       nan        nan 0.50306122 0.51072523        nan] 평균 : nan
ClassifierChain 은 없는 놈!!!
ComplementNB [0.38163265 0.37653061 0.36632653 0.34320735 0.36159346] 평균 : 0.3659
DecisionTreeClassifier [0.62959184 0.59081633 0.59285714 0.57609806 0.60572012] 평균 : 0.599
DummyClassifier [0.45816327 0.43367347 0.44285714 0.46271706 0.44637385] 평균 : 0.4488
ExtraTreeClassifier [0.60918367 0.59387755 0.60204082 0.59346272 0.62104188] 평균 : 0.6039
ExtraTreesClassifier [0.71530612 0.66428571 0.7        0.70684372 0.69662921] 평균 : 0.6966
GaussianNB [0.46530612 0.44591837 0.45510204 0.41266599 0.46373851] 평균 : 0.4485
GaussianProcessClassifier [0.59693878 0.57244898 0.58163265 0.57405516 0.57099081] 평균 : 0.5792
GradientBoostingClassifier [0.61122449 0.57244898 0.59489796 0.61287028 0.59244127] 평균 : 0.5968
HistGradientBoostingClassifier [0.69183673 0.6622449  0.67653061 0.67109295 0.6639428 ] 평균 : 0.6731
KNeighborsClassifier [0.48979592 0.48469388 0.4755102  0.46373851 0.45863126] 평균 : 0.4745
LabelPropagation [0.59387755 0.57244898 0.57040816 0.5628192  0.56588355] 평균 : 0.5731
LabelSpreading [0.59387755 0.57244898 0.57040816 0.56384065 0.56588355] 평균 : 0.5733
LinearDiscriminantAnalysis [0.5255102  0.51326531 0.5244898  0.56384065 0.52706844] 평균 : 0.5308
LinearSVC [0.45408163 0.46020408 0.4        0.46475996 0.35035751] 평균 : 0.4259
LogisticRegression [0.47142857 0.45204082 0.44795918 0.48723187 0.46578141] 평균 : 0.4649
LogisticRegressionCV [0.50204082 0.49591837 0.48979592 0.53421859 0.49233912] 평균 : 0.5029
MLPClassifier [0.53367347 0.48367347 0.4744898  0.52502554 0.50153218] 평균 : 0.5037
MultiOutputClassifier 은 없는 놈!!!
NearestCentroid [0.12959184 0.10204082 0.10102041 0.11235955 0.09090909] 평균 : 0.1072
NuSVC [nan nan nan nan nan] 평균 : nan
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier [0.45714286 0.43367347 0.46326531 0.46271706 0.4494382 ] 평균 : 0.4532
Perceptron [0.45816327 0.43367347 0.32244898 0.33094995 0.09499489] 평균 : 0.328
QuadraticDiscriminantAnalysis [0.48367347 0.45102041 0.50306122 0.46782431 0.48008172] 평균 : 0.4771
RadiusNeighborsClassifier [nan nan nan nan nan] 평균 : nan
RandomForestClassifier [0.70816327 0.67346939 0.69081633 0.70275792 0.68947906] 평균 : 0.6929
RidgeClassifier [0.53163265 0.5122449  0.52142857 0.54954035 0.51276813] 평균 : 0.5255
RidgeClassifierCV [0.53163265 0.5122449  0.52142857 0.54954035 0.51276813] 평균 : 0.5255
SGDClassifier [0.45816327 0.33673469 0.47959184 0.37282942 0.18488253] 평균 : 0.3664
SVC [0.4622449  0.4377551  0.44693878 0.46373851 0.4473953 ] 평균 : 0.4516
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
'''