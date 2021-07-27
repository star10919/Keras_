from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np
from icecream import ic
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

### 넘파이로 데이터 저장 - csv 로 불러오는 거보다 시간 절약 됨

# 사이킷런 데이터셋 넘파이로 저장
datasets1 = load_iris()

x_data_iris = datasets1.data
y_data_iris = datasets1.target

# ic(type(x_data), type(y_data))      # type(x_data): <class 'numpy.ndarray'>, type(y_data): <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris)


datasets2 = load_boston()
x_data_boston = datasets2.data
y_data_boston = datasets2.target
np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data_boston)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data_boston)


datasets3 = load_breast_cancer()
x_data_cancer = datasets3.data
y_data_cancer = datasets3.target
np.save('./_save/_npy/k55_x_data_cancer.npy', arr=x_data_cancer)
np.save('./_save/_npy/k55_y_data_cancer.npy', arr=y_data_cancer)


datasets4 = load_diabetes()
x_data_diabet = datasets4.data
y_data_diabet = datasets4.target
np.save('./_save/_npy/k55_x_data_diabet.npy', arr=x_data_diabet)
np.save('./_save/_npy/k55_y_data_diabet.npy', arr=y_data_diabet)


datasets5 = load_wine()
x_data_wine = datasets5.data
y_data_wine = datasets5.target
np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)




# 텐서플로우 데이터셋 넘파이로 저장
(x_train_minst, y_train_mnist), (x_test_minst, y_test_minst) = mnist.load_data()
np.save('./_save/_npy/k55_x_train_mnist.npy', arr=x_train_minst)
np.save('./_save/_npy/k55_x_test_mnist.npy', arr=x_test_minst)
np.save('./_save/_npy/k55_y_train_mnist.npy', arr=y_train_mnist)
np.save('./_save/_npy/k55_y_test_mnist.npy', arr=y_test_minst)


(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()
np.save('./_save/_npy/k55_x_train_fashtion.npy', arr=x_train_fashion)
np.save('./_save/_npy/k55_x_test_fashtion.npy', arr=x_test_fashion)
np.save('./_save/_npy/k55_y_train_fashtion.npy', arr=y_train_fashion)
np.save('./_save/_npy/k55_y_test_fashtion.npy', arr=y_test_fashion)


(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()
np.save('./_save/_npy/k55_x_train_cifar10.npy', arr=x_train_cifar10)
np.save('./_save/_npy/k55_x_test_cifar10.npy', arr=x_test_cifar10)
np.save('./_save/_npy/k55_y_train_cifar10.npy', arr=y_train_cifar10)
np.save('./_save/_npy/k55_y_test_cifar10.npy', arr=y_test_cifar10)


(x_train_cifar100, y_train_cifar100), (x_test_cifar100, y_test_cifar100) = cifar100.load_data()
np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=x_train_cifar100)
np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=x_test_cifar100)
np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=y_train_cifar100)
np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=y_test_cifar100)