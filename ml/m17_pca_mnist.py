import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from icecream import ic
(x_train, _), (x_test, _) = mnist.load_data()       # _ : 변수로 받지 않겠다!

ic(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
ic(x.shape)      # x.shape: (70000, 28, 28)




# 실습
# pca를 통해 0.95 이상인 n_components 가 몇 개?
# 3차원이 들어가는지? 안 들어가면 reshape해줘야 함  -  PCA는 2차원으로 받아줘야 함!!!!!!!!


x = x.reshape(x.shape[0], 28*28)
ic(x.shape)     # (70000, 784)


pca = PCA(n_components=154)
x = pca.fit_transform(x)
ic(x)
ic(x.shape)                # (70000, 784)

pca_EVR = pca.explained_variance_ratio_     # 피쳐임포턴스가 낮은순서대로 압축됨. 피쳐임포턴스 높은순서대로 보여줌
ic(pca_EVR)
ic(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)     # 피쳐 임포턴스 낮은거부터 누적해서 보여줌
ic(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]

ic(np.argmax(cumsum >= 0.95)+1) # 154      / 여기서 나온 수를 n_components에 넣어 주면 됨

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
