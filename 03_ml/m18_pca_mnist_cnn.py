import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from icecream import ic

# 실습
# mnist 데이터를 pca를 통해 cnn으로 구성
# (28, 28) -> 784 -> PCA(400) -> (20, 20) -> CNN 모델 구성

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()       # _ : 변수로 받지 않겠다!

ic(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
ic(y_train.shape, y_test.shape)      # (60000,) (10000,)

x = np.append(x_train, x_test, axis=0)
ic(x.shape)      # x.shape: (70000, 28, 28)

y= np.append(y_train, y_test, axis=0)
ic(y.shape)      # y.shape: (70000,)
ic(np.unique(y))    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

x = x.reshape(x.shape[0], 28*28)
ic(x.shape)     # (70000, 784)




pca = PCA(n_components=400)  # 전체컬럼수 넣고, 아그맥스에서 추출된 값 넣어서 돌리기
x = pca.fit_transform(x)
ic(x)
ic(x.shape)                # (70000, 400)


# pca_EVR = pca.explained_variance_ratio_     # 피쳐임포턴스가 낮은순서대로 압축됨. 피쳐임포턴스 높은순서대로 보여줌
# ic(pca_EVR)
# ic(sum(pca_EVR))

# cumsum = np.cumsum(pca_EVR)     # 피쳐 임포턴스 낮은거부터 누적해서 보여줌
# ic(cumsum)
# # [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]

# ic(np.argmax(cumsum >= 0.999)+1) # 486      / 여기서 나온 수를 n_components에 넣어 주면 됨

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x = x.reshape(x.shape[0], 20, 20, 1)



# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
# <DNN>
# model = Sequential()
# model.add(Dense(10, input_shape=(486,)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# <CNN>
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(20, 20, 1), padding='same', activation='relu'))
model.add(Conv2D(10, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(10, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))



# 3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['acc'])

model.fit(x, y, epochs=100, batch_size=500)



# 4. 평가
results = model.evaluate(x, y)
print("acc :", results[1])


'''
<DNN>
* 전체컬럼
acc : 0.8911285996437073

* PCA(486)
acc : 0.9537285566329956



<CNN>
* 전체컬럼
acc : 1.0

* PCA(486)
acc : 1.0

* PCA(400)
acc : 0.9665714502334595
'''