import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, LSTM, Conv1D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.activations import linear

# 2진 분류

# 1. 데이터
datasets = load_breast_cancer()
# ic(datasets.DESCR)
# ic(datasets.feature_names)

x = datasets.data
y = datasets.target  #2진 분류(0 or 1)

# ic(x.shape, y.shape)   #(569, 30), (569,)
# ic(y[:40])  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# ic(np.unique(y))   # [0, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9)

# 1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   # x_train.shape: (398, 30), x_test.shape: (171, 30)
# x_train = x_train.reshape(398, 30, 1)
# x_test = x_test.reshape(171, 30, 1)


# 2. 모델
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
AdaBoostClassifier 의 정답률 : 0.9649122807017544
BaggingClassifier 의 정답률 : 0.9649122807017544
BernoulliNB 의 정답률 : 0.9473684210526315
CalibratedClassifierCV 의 정답률 : 0.9590643274853801
CategoricalNB 은 없는 놈!!!
ClassifierChain 은 없는 놈!!!
ComplementNB 은 없는 놈!!!
DecisionTreeClassifier 의 정답률 : 0.9239766081871345
DummyClassifier 의 정답률 : 0.6374269005847953
ExtraTreeClassifier 의 정답률 : 0.9473684210526315
ExtraTreesClassifier 의 정답률 : 0.9707602339181286
GaussianNB 의 정답률 : 0.9298245614035088
GaussianProcessClassifier 의 정답률 : 0.9766081871345029
GradientBoostingClassifier 의 정답률 : 0.9532163742690059
HistGradientBoostingClassifier 의 정답률 : 0.9707602339181286
KNeighborsClassifier 의 정답률 : 0.9707602339181286
LabelPropagation 의 정답률 : 0.9707602339181286
LabelSpreading 의 정답률 : 0.9707602339181286
LinearDiscriminantAnalysis 의 정답률 : 0.9649122807017544
LinearSVC 의 정답률 : 0.9824561403508771
LogisticRegression 의 정답률 : 0.9883040935672515
LogisticRegressionCV 의 정답률 : 0.9883040935672515
MLPClassifier 의 정답률 : 0.9824561403508771 
MultiOutputClassifier 은 없는 놈!!!
MultinomialNB 은 없는 놈!!!
NearestCentroid 의 정답률 : 0.935672514619883
NuSVC 의 정답률 : 0.9649122807017544
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.9766081871345029
Perceptron 의 정답률 : 0.9766081871345029
QuadraticDiscriminantAnalysis 의 정답률 : 0.9590643274853801
RadiusNeighborsClassifier 은 없는 놈!!!
RandomForestClassifier 의 정답률 : 0.9590643274853801
RidgeClassifier 의 정답률 : 0.9649122807017544
RidgeClassifierCV 의 정답률 : 0.9649122807017544
SGDClassifier 의 정답률 : 0.9883040935672515
SVC 의 정답률 : 0.9883040935672515
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
'''