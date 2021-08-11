import warnings
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic
from tensorflow.python.keras.layers.core import Dropout

### all_estimators(모든 알고리즘(모델) 사용)
# 실습, 모델구성하고 완료하시오.
# 회귀 데이터를 Classifier로 만들었을 경우에 에러 확인!!!

# 클래시파이가 아니므로 에러가 날거임


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import KFold, cross_val_score

#2. 모델
# from sklearn.utils.testing import all_estimators
from sklearn.utils import all_estimators        # 이걸로 바뀜(testing 빠짐)
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')               ### warning 무시

# allAlgorithms = all_estimators(type_filter='classifier')      # 분류모델
allAlgorithms = all_estimators(type_filter='regressor')         # 회귀모델
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
모델의 개수 : 54
ARDRegression [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 평균 : 0.6985
AdaBoostRegressor [0.90733113 0.82466501 0.78621947 0.81130226 0.86355343] 평균 : 0.8386
BaggingRegressor [0.9051967  0.85630531 0.83831306 0.86077471 0.87130209] 평균 : 0.8664
BayesianRidge [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 평균 : 0.7038
CCA [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276] 평균 : 0.6471
DecisionTreeRegressor [0.80101718 0.55200115 0.79249099 0.73151884 0.78589282] 평균 : 0.7326
DummyRegressor [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 : -0.0135
ElasticNet [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354] 평균 : 0.6708
ElasticNetCV [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 평균 : 0.6565
ExtraTreeRegressor [0.8005609  0.60435102 0.43928988 0.72383045 0.84149269] 평균 : 0.6819
ExtraTreesRegressor [0.93471749 0.85511183 0.77353627 0.87563569 0.93626706] 평균 : 0.8751
GammaRegressor [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635] 평균 : -0.0136
GaussianProcessRegressor [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828] 평균 : -5.9286
GradientBoostingRegressor [0.94534463 0.83655439 0.82718127 0.88757488 0.93236063] 평균 : 0.8858
HistGradientBoostingRegressor [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 평균 : 0.8581
HuberRegressor [0.74400323 0.64244715 0.52848946 0.37100122 0.63403398] 평균 : 0.584
IsotonicRegression [nan nan nan nan nan] 평균 : nan
KNeighborsRegressor [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 평균 : 0.5286
KernelRidge [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 평균 : 0.6854
Lars [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 평균 : 0.6977
LarsCV [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 평균 : 0.6928
Lasso [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473] 평균 : 0.6657
LassoCV [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 평균 : 0.6779
LassoLars [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 : -0.0135
LassoLarsCV [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 평균 : 0.6965
LassoLarsIC [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 평균 : 0.713
LinearRegression [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 : 0.7128
LinearSVR [0.68715273 0.58258982 0.43544275 0.51720939 0.64692245] 평균 : 0.5739
MLPRegressor [0.60206701 0.54877603 0.3028612  0.46935967 0.56681928] 평균 : 0.498
MultiOutputRegressor 은 없는 놈!!!
MultiTaskElasticNet [nan nan nan nan nan] 평균 : nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 : nan
MultiTaskLasso [nan nan nan nan nan] 평균 : nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 : nan
NuSVR [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 평균 : 0.2295
OrthogonalMatchingPursuit [0.58276176 0.565867   0.48689774 0.51545117 0.52049576] 평균 : 0.5343
OrthogonalMatchingPursuitCV [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 평균 : 0.6578
PLSCanonical [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868] 평균 : -2.2096
PLSRegression [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 평균 : 0.6847
PassiveAggressiveRegressor [ 0.05125203  0.26102024 -0.06460685  0.1617814   0.27249374] 평균 : 0.1364
PoissonRegressor [0.85659255 0.8189989  0.66691488 0.67998192 0.75195656] 평균 : 0.7549
RANSACRegressor [0.4625533  0.66915321 0.56522477 0.35340269 0.64529627] 평균 : 0.5391
RadiusNeighborsRegressor [nan nan nan nan nan] 평균 : nan
RandomForestRegressor [0.91994096 0.85843122 0.81921569 0.88881305 0.89996103] 평균 : 0.8773
RegressorChain 은 없는 놈!!!
Ridge [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776] 평균 : 0.7109
RidgeCV [0.81125292 0.80010535 0.58888304 0.64008984 0.72362912] 평균 : 0.7128
SGDRegressor [-2.19450995e+26 -1.72569033e+26 -8.10651252e+25 -1.02705257e+27
 -2.32882168e+26] 평균 : -3.466039782985032e+26
SVR [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 평균 : 0.1963
StackingRegressor 은 없는 놈!!!
TheilSenRegressor [0.78642061 0.72028197 0.58591645 0.55835444 0.72472201] 평균 : 0.6751
TransformedTargetRegressor [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 : 0.7128
TweedieRegressor [0.7492543  0.75457294 0.56286929 0.57989884 0.63242475] 평균 : 0.6558
VotingRegressor 은 없는 놈!!!
'''