from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from icecream import ic
import numpy as np
from tensorflow.python.ops.gen_dataset_ops import batch_dataset

### EarlyStopping

# standardization : 표준정규분포 에 맞춰서 scaler 함

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# ic(np.min(x), np.max(x))   # np.min(x): 0.0, np.max(x): 711.0

# 데이터 전처리
# 부동소수점 사용(정수보다는 소수점끼리의 연산이 더 빠르니까) 후 X100   <= 데이터 전처리(반드시 해야 함)
# (방법 1)x = x/711.
# (방법 2)x = x/np.max(x)
# (방법 3)x = (x - np.min(x)) / (np.max(x) - np.min(x))     # x = (x - min) / (max - min)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=9)
# ic(x_test)
# ic(y_test)

# (방법 4)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)   ##1- 실행만 시킨거임   / train만을 fit 시켜서(전체 데이터로 minmaxscaler하면 과적합 됨)
x_train = scaler.transform(x_train)   ##2- 변환(scaler됨)   / train 기준에 스케일된 걸로 test transfrom해줌
x_test = scaler.transform(x_test)
# x_pred = scaler.transform(x_pred)
   #  =>test 데이터는 train 데이터에 반영되면 안된다!!!!!!!!!!!!!!!!!!!



ic(x.shape, x_train.shape, x_test.shape)   # (506, 13) (404, 13) (102, 13)
ic(y.shape, y_train.shape, y_test.shape)   # (506,) (404,) (102,)

# train_test_split(70%) 사용,       loss 값, r2(0.7정도는 넘기기) 값 출력

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


#3. 컴파일(EarlyStopping), 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)   # patience=5  loss값이 5번이 지날때까지 참겠다.   # mode='min'  최솟값이 나올때까지
# EarlyStopping은 verbose 안쓰는 경우가 더 많음  /  verbose=1 -> Epoch 00042: early stopping 추가

hist = model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.3, callbacks=[es])   # 컴파일에 입력한거 callbacks파리미터 입력해서 적용시켜주기


# ic(hist.history.keys())  # dict_keys(['loss', 'val_loss'])
ic(hist.history['loss'])
ic(hist.history['val_loss'])

'''
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])   # x:epoch / y:hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.title("loss, val_loss")
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.show()
'''


#4. 평가, 예측, r2결정계수
loss = model.evaluate(x_test, y_test)   # batch_size=32(디폴트 값)
ic(loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
ic(r2)


# 그래프 그리기
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'gulim'   # 한글 깨짐 방지
plt.plot(hist.history['loss'])   # x:epoch / y:hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.title("로스, 발로스")
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train loss', 'val loss'])   # 범례
plt.show()


'''
#5. 결과값
ic| loss: 6.996442794799805
ic| r2: 0.9308562809262234
'''


