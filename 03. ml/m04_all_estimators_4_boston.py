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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=5)
# ic(x_test)
# ic(y_test)

# ic(x.shape, x_train.shape, x_test.shape)   # x.shape: (506, 13), x_train.shape: (404, 13), x_test.shape: (102, 13)
# ic(y.shape, y_train.shape, y_test.shape)   # y.shape: (506,), y_train.shape: (404,), y_test.shape: (102,)

#1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   # x_train.shape: (354, 13), x_test.shape: (152, 13)
# x_train = x_train.reshape(354, 13, 1)
# x_test = x_test.reshape(152, 13, 1)



#2. 모델
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
ARDRegression 의 정답률 : 0.7012854759131435
AdaBoostRegressor 의 정답률 : 0.8604613724460279
BaggingRegressor 의 정답률 : 0.8737516444286395
BayesianRidge 의 정답률 : 0.7155571091660405
CCA 의 정답률 : 0.7032720871825833
DecisionTreeRegressor 의 정답률 : 0.7868667449040481
DummyRegressor 의 정답률 : -6.386072533048903e-05
ElasticNet 의 정답률 : 0.23493686144495995
ElasticNetCV 의 정답률 : 0.7089968460721139
ExtraTreeRegressor 의 정답률 : 0.7542540295969593
ExtraTreesRegressor 의 정답률 : 0.8865516995063676
GammaRegressor 의 정답률 : 0.21773908282362264
GaussianProcessRegressor 의 정답률 : 0.5080239369196433
GradientBoostingRegressor 의 정답률 : 0.9070301624144191
HistGradientBoostingRegressor 의 정답률 : 0.8572787852888598
HuberRegressor 의 정답률 : 0.6811548709020908
IsotonicRegression 은 없는 놈!!!
KNeighborsRegressor 의 정답률 : 0.6536005738680479
KernelRidge 의 정답률 : 0.46169092151630353
Lars 의 정답률 : 0.7243621768769086
LarsCV 의 정답률 : 0.7074612071066502
Lasso 의 정답률 : 0.44660762377435337
LassoCV 의 정답률 : 0.7066579626634952
LassoLars 의 정답률 : -6.386072533048903e-05
LassoLarsCV 의 정답률 : 0.7074612071066502
LassoLarsIC 의 정답률 : 0.6574158223432565
LinearRegression 의 정답률 : 0.7243621768769084
LinearSVR 의 정답률 : 0.5276136574875525
MLPRegressor 의 정답률 : 0.1697877708435397
MultiOutputRegressor 은 없는 놈!!!
MultiTaskElasticNet 은 없는 놈!!!
MultiTaskElasticNetCV 은 없는 놈!!!
MultiTaskLasso 은 없는 놈!!!
MultiTaskLassoCV 은 없는 놈!!!
NuSVR 의 정답률 : 0.5119288922971168
OrthogonalMatchingPursuit 의 정답률 : 0.6238888779636274
OrthogonalMatchingPursuitCV 의 정답률 : 0.6958727170166868
PLSCanonical 의 정답률 : -2.535922781187592
PLSRegression 의 정답률 : 0.6428033767102312
PassiveAggressiveRegressor 의 정답률 : 0.6794814842673622
PoissonRegressor 의 정답률 : 0.6074982788336136
RANSACRegressor 의 정답률 : 0.5271698283977824
RadiusNeighborsRegressor 의 정답률 : 0.5255304392687142
RandomForestRegressor 의 정답률 : 0.8672612149563234
RegressorChain 은 없는 놈!!!
Ridge 의 정답률 : 0.7095948409438191
RidgeCV 의 정답률 : 0.7095948409438262
SGDRegressor 의 정답률 : 0.6769383658381725
SVR 의 정답률 : 0.5290228507623143
StackingRegressor 은 없는 놈!!!
TheilSenRegressor 의 정답률 : 0.6750761678176758
TransformedTargetRegressor 의 정답률 : 0.7243621768769084
TweedieRegressor 의 정답률 : 0.22217055128772256
VotingRegressor 은 없는 놈!!!
'''