import warnings
import numpy as np
import pandas as pd
from icecream import ic
from pandas.core.tools.datetimes import Scalar
from tensorflow.python.keras.layers.recurrent import LSTM

# 다중분류
# 모델링하고
# 0.8 이상 완성


# 1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',       # 경로잡기 중요!
                        index_col=None, header=0)    #header=0 첫번째라인   # (4898,12)

# * visual studio 기준(파이참이랑 다름)
    # ./  :  현재폴더(STUDY)
    # ../ :  상위폴더

# print(datasets)   # quality 가  y

# 아래 3개는 꼭 찍어보기
    # ic(datasets.shape)   # (4898, 12)   => x:(4898,11),   y:(4898,) 으로 잘라주기
    # ic(datasets.info())
    # ic(datasets.describe())


#1 판다스 -> 넘파이 : index와 header가 날아감
#2 x와 y를 분리
#3. sklearn의 OneHotEncoder 사용할것
#3 y의 라벨을 확인 np.unique(y) <= Output node 개수 잡아주기 위해서
#5. y의 shape 확인 (4898,) -> (4898,7)


datasets_np = datasets.to_numpy()   #1 판다스 -> 넘파이
ic(datasets_np)
x = datasets_np[:,0:11]
ic(x)
y = datasets_np[:,[-1]]
ic(y)
ic(x.shape, y.shape)   # x.shape: (4898, 11), y.shape: (4898,1)
ic(np.unique(y))   # [3, 4, 5, 6, 7, 8, 9]  -  7개


# from sklearn.preprocessing import OneHotEncoder    # 0, 1, 2 자동 채움 안됨 / # to_categorical 0, 1, 2 없으나 자동 생성
# onehot = OneHotEncoder()
# onehot.fit(y)
# y = onehot.transform(y).toarray() 
# ic(y.shape)    # (4898, 7)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.995, shuffle=True, random_state=24)

# x 데이터 전처리(scaler)
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   #  x_train.shape: (4873, 11), x_test.shape: (25, 11)
# x_train = x_train.reshape(4873, 11, 1)
# x_test = x_test.reshape(25, 11, 1)


# 2. 모델 구성
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
AdaBoostClassifier 의 정답률 : 0.48
BaggingClassifier 의 정답률 : 0.72
BernoulliNB 의 정답률 : 0.56
CalibratedClassifierCV 의 정답률 : 0.56
CategoricalNB 은 없는 놈!!!
ClassifierChain 은 없는 놈!!!
ComplementNB 은 없는 놈!!!
DecisionTreeClassifier 의 정답률 : 0.56
DummyClassifier 의 정답률 : 0.4
ExtraTreeClassifier 의 정답률 : 0.8
ExtraTreesClassifier 의 정답률 : 0.72
GaussianNB 의 정답률 : 0.52
GaussianProcessClassifier 의 정답률 : 0.72
GradientBoostingClassifier 의 정답률 : 0.64
HistGradientBoostingClassifier 의 정답률 : 0.8
KNeighborsClassifier 의 정답률 : 0.72
LabelPropagation 의 정답률 : 0.64
LabelSpreading 의 정답률 : 0.64
LinearDiscriminantAnalysis 의 정답률 : 0.44
LinearSVC 의 정답률 : 0.6
LogisticRegression 의 정답률 : 0.52
LogisticRegressionCV 의 정답률 : 0.52
MLPClassifier 의 정답률 : 0.68
MultiOutputClassifier 은 없는 놈!!!
MultinomialNB 은 없는 놈!!!
NearestCentroid 의 정답률 : 0.24
NuSVC 은 없는 놈!!!
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.56
Perceptron 의 정답률 : 0.44
QuadraticDiscriminantAnalysis 의 정답률 : 0.44
RadiusNeighborsClassifier 은 없는 놈!!!
RandomForestClassifier 의 정답률 : 0.76
RidgeClassifier 의 정답률 : 0.6
RidgeClassifierCV 의 정답률 : 0.6
SGDClassifier 의 정답률 : 0.52
SVC 의 정답률 : 0.64
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
'''