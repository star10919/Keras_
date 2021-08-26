# DNN
# sigmid, linear
# 단층 퍼셉트론으로 구성
# sigmoid


# from tensorflow.keras.datasets import mnist     # 노란줄은 나오지만 먹히긴 먹힘
from keras.datasets import mnist        # pip install keras==2.3.1

import tensorflow as tf
import numpy as np

tf.set_random_seed(66)


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# 아웃풋레이어
w = tf.compat.v1.Variable(tf.random.normal([28*28, 10]), name='weight3')      # 아웃풋 열은 최종 열로 맞춰줘야 함
b = tf.compat.v1.Variable(tf.random.normal([10]), name='bias3')  

# hypothesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)    # sigmoid



# cost = tf.reduce_mean(tf.square(hypothesis-y))      # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy :  y 값이 0과 1 사이로 바꼈으니까

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)

train = optimizer.minimize(cost)
