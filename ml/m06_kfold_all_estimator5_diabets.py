from operator import mod
import warnings
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

### all_estimators(모든 알고리즘(모델) 사용)
# 실습, 모델구성하고 완료하시오.
# 회귀 데이터를 Classifier로 만들었을 경우에 에러 확인!!!

# 클래시파이가 아니므로 에러가 날거임


#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target


from sklearn.model_selection import KFold, cross_val_score

#2. 모델구성(validation)
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
ARDRegression [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369] 평균 : 0.4923
AdaBoostRegressor [0.34763413 0.48143215 0.50559323 0.38268943 0.42353703] 평균 : 0.4282
BaggingRegressor [0.34614974 0.48254994 0.46672822 0.29940893 0.40154217] 평균 : 0.3993
BayesianRidge [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ] 평균 : 0.4893
CCA [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701] 평균 : 0.438
DecisionTreeRegressor [-0.2293478  -0.14856571 -0.05683057  0.02092888  0.0103865 ] 평균 : -0.0807
DummyRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 : -0.0033
ElasticNet [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988] 평균 : 0.0054
ElasticNetCV [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ] 평균 : 0.4394
ExtraTreeRegressor [-0.06279527  0.16132114  0.08084632  0.33476471 -0.20833024] 평균 : 0.0612
ExtraTreesRegressor [0.3957392  0.48048377 0.5282528  0.40262955 0.45284766] 평균 : 0.452
GammaRegressor [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898] 평균 : 0.0027
GaussianProcessRegressor [ -5.6360757  -15.27401119  -9.94981439 -12.46884878 -12.04794389] 평균 : -11.0753
GradientBoostingRegressor [0.39008659 0.4811443  0.48023242 0.3931337  0.4437622 ] 평균 : 0.4377
HistGradientBoostingRegressor [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755] 평균 : 0.3947
HuberRegressor [0.50334705 0.47508237 0.54650576 0.36883712 0.5173073 ] 평균 : 0.4822
IsotonicRegression [nan nan nan nan nan] 평균 : nan
KNeighborsRegressor [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 평균 : 0.3673
KernelRidge [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537] 평균 : -3.5938
Lars [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679] 평균 : -0.1495
LarsCV [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596] 평균 : 0.4879
Lasso [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ] 평균 : 0.3518
LassoCV [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393] 평균 : 0.487
LassoLars [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891] 평균 : 0.3742
LassoLarsCV [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679] 평균 : 0.4866
LassoLarsIC [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ] 평균 : 0.4912
LinearRegression [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 : 0.4876
LinearSVR [-0.33470258 -0.31629592 -0.41913109 -0.30164983 -0.4677611 ] 평균 : -0.3679
MLPRegressor [-2.6854452  -2.97194635 -3.38076128 -2.77517237 -3.30474645] 평균 : -3.0236
MultiOutputRegressor 은 없는 놈!!!
MultiTaskElasticNet [nan nan nan nan nan] 평균 : nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 : nan
MultiTaskLasso [nan nan nan nan nan] 평균 : nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 : nan
NuSVR [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ] 평균 : 0.1618
OrthogonalMatchingPursuit [0.32934491 0.285747   0.38943221 0.19671679 0.35916077] 평균 : 0.3121
OrthogonalMatchingPursuitCV [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516] 평균 : 0.4857
PLSCanonical [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996] 평균 : -1.2086
PLSRegression [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873] 평균 : 0.4842
PassiveAggressiveRegressor [0.41927438 0.46835878 0.52293542 0.35975378 0.51575854] 평균 : 0.4572
PoissonRegressor [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626] 평균 : 0.3341
RANSACRegressor [-0.0171573   0.21684949  0.36472494 -0.00249008  0.28301837] 평균 : 0.169
RadiusNeighborsRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 : -0.0033
RandomForestRegressor [0.36715275 0.49816114 0.48178451 0.35977319 0.45188699] 평균 : 0.4318
RegressorChain 은 없는 놈!!!
Ridge [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091] 평균 : 0.4212
RidgeCV [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194] 평균 : 0.4884
SGDRegressor [0.39328882 0.44185719 0.46468524 0.32964328 0.4152881 ] 평균 : 0.409
SVR [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ] 평균 : 0.1591
StackingRegressor 은 없는 놈!!!
TheilSenRegressor [0.51081382 0.45155128 0.54814958 0.34938764 0.52796915] 평균 : 0.4776
TransformedTargetRegressor [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 : 0.4876
TweedieRegressor [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042] 평균 : 0.0032
VotingRegressor 은 없는 놈!!!
'''