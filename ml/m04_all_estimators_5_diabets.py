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

# ic(x.shape, y.shape)  # (442, 10)  (442,)

# ic(datasets.feature_names)   
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

# ic(x[:30])
# ic(np.min(y), np.max(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

# x 데이터 전처리
from sklearn.preprocessing import StandardScaler, PowerTransformer, MaxAbsScaler
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   # x_train.shape: (353, 10), x_test.shape: (89, 10)

# x_train = x_train.reshape(353, 10, 1)
# x_test = x_test.reshape(89, 10, 1)

#2. 모델구성(validation)
# from sklearn.utils.testing import all_estimators
from sklearn.utils import all_estimators        # 이걸로 바뀜(testing 빠짐)
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')               ### warning 무시

# allAlgorithms = all_estimators(type_filter='classifier')      # 분류모델
allAlgorithms = all_estimators(type_filter='regressor')         # 회귀모델
# ic(allAlgorithms)
print('모델의 개수 :', len(allAlgorithms))      # 모델의 개수 : 41


for ( name, algorithm ) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)
        # print(name)

        y_predict = model.predict(x_test)   #score 안먹히는 거 있어서    predict 사용
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률 :', r2)
    
    except:         
        # continue      # 에러뜨는거 제외하고 정상적인 거만 돌아감
        print(name, "은 없는 놈!!!")

'''
모델의 개수 : 54
ARDRegression 의 정답률 : 0.5894082234364912
AdaBoostRegressor 의 정답률 : 0.5214729797676871
BaggingRegressor 의 정답률 : 0.4374764232871976
BayesianRidge 의 정답률 : 0.5939553828751604
CCA 의 정답률 : 0.5852713803269576
DecisionTreeRegressor 의 정답률 : -0.17050413212749027
DummyRegressor 의 정답률 : -0.01545589029660177
ElasticNet 의 정답률 : 0.3033226500383236
ElasticNetCV 의 정답률 : 0.5859828130931419
ExtraTreeRegressor 의 정답률 : -0.10386234528271587
ExtraTreesRegressor 의 정답률 : 0.5617802250421415
GammaRegressor 의 정답률 : 0.21477056070970568
GaussianProcessRegressor 의 정답률 : -4.720679276332729
GradientBoostingRegressor 의 정답률 : 0.551332872074566
HistGradientBoostingRegressor 의 정답률 : 0.5390878373341259
HuberRegressor 의 정답률 : 0.5775590274753332
IsotonicRegression 은 없는 놈!!!
KNeighborsRegressor 의 정답률 : 0.48703039459904796
KernelRidge 의 정답률 : -3.5006848395873957
Lars 의 정답률 : 0.5851141269959728
LarsCV 의 정답률 : 0.5896206660679896
Lasso 의 정답률 : 0.5834814546406213
LassoCV 의 정답률 : 0.5842905370893492
LassoLars 의 정답률 : 0.45238726393914497
LassoLarsCV 의 정답률 : 0.5896206660679896
LassoLarsIC 의 정답률 : 0.5962837270477234
LinearRegression 의 정답률 : 0.5851141269959736
LinearSVR 의 정답률 : 0.2241656063577484
MLPRegressor 의 정답률 : -2.001293705371808
MultiOutputRegressor 은 없는 놈!!!
MultiTaskElasticNet 은 없는 놈!!!
MultiTaskElasticNetCV 은 없는 놈!!!
MultiTaskLasso 은 없는 놈!!!
MultiTaskLassoCV 은 없는 놈!!!
NuSVR 의 정답률 : 0.15915237551458283
OrthogonalMatchingPursuit 의 정답률 : 0.32241768669099435
OrthogonalMatchingPursuitCV 의 정답률 : 0.581211443040667
PLSCanonical 의 정답률 : -1.687851891160105
PLSRegression 의 정답률 : 0.6072433305368464
PassiveAggressiveRegressor 의 정답률 : 0.5690462204251914
PoissonRegressor 의 정답률 : 0.5828337809520332
RANSACRegressor 의 정답률 : 0.43417722697331473
RadiusNeighborsRegressor 의 정답률 : 0.5220800156268794
RandomForestRegressor 의 정답률 : 0.549244334051673
RegressorChain 은 없는 놈!!!
Ridge 의 정답률 : 0.5946260113292288
RidgeCV 의 정답률 : 0.5899471919176922
SGDRegressor 의 정답률 : 0.5926716794578409
SVR 의 정답률 : 0.18125655353639103
StackingRegressor 은 없는 놈!!!
TheilSenRegressor 의 정답률 : 0.5918269528708773
TransformedTargetRegressor 의 정답률 : 0.5851141269959736
TweedieRegressor 의 정답률 : 0.2078653839675212
VotingRegressor 은 없는 놈!!!
'''