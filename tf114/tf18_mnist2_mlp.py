# DNN
# sigmid, linear
# 다층 퍼셉트론(multi layer perceptron)으로 구성
# accuracy 넣기 : 0.97 넘기
### acticate


# from tensorflow.keras.datasets import mnist     # 노란줄은 나오지만 먹히긴 먹힘
from keras.datasets import mnist        # pip install keras==2.3.1

import tensorflow as tf
import numpy as np

tf.set_random_seed(66)


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  #  (10000, 28, 28) (10000,)

x_train = x_train.reshape(x_train.shape[0], 28*28)/255.
x_test = x_test.reshape(x_test.shape[0], 28*28)/255.
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

print(x_train.shape, y_train.shape)  # (60000, 784) (60000, 1)
print(x_test.shape, y_test.shape)    # (10000, 784) (10000, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()      # toarray : list자료형태로 바꿔줌
y_test = one.transform(y_test).toarray()
print(y_train, y_train.shape)   # (60000, 10)
print(y_test, y_test.shape)     # (10000, 10)




# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# 히든레이어1
w1 = tf.compat.v1.Variable(tf.random.normal([28*28, 512]), name='weight1')      # 아웃풋 열은 최종 열로 맞춰줘야 함
b1 = tf.compat.v1.Variable(tf.random.normal([512]), name='bias1')  

layers1 = tf.sigmoid(tf.matmul(x, w1) + b1)    # sigmoid



# 히든레이어 2
w2 = tf.compat.v1.Variable(tf.random.normal([512,256]), name='weight2')       # [윗 레이어의 열, 내가 주고 싶은 노드의 개수]
b2 = tf.compat.v1.Variable(tf.random.normal([256]), name='bias2')                     

# hypothesis = x * w + b
layers2 = tf.sigmoid(tf.matmul(layers1, w2) + b2)    # sigmoid



# 히든레이어 3
w3 = tf.compat.v1.Variable(tf.random.normal([256,46]), name='weight2')       # [윗 레이어의 열, 내가 주고 싶은 노드의 개수]
b3 = tf.compat.v1.Variable(tf.random.normal([46]), name='bias2')                     

# hypothesis = x * w + b
layers3 = tf.sigmoid(tf.matmul(layers2, w3) + b3)    # sigmoid



# 아웃풋레이어
w4 = tf.compat.v1.Variable(tf.random.normal([46,10]), name='weight3')      # 아웃풋 열은 최종 열로 맞춰줘야 함
b4 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias3')  

# hypothesis = x * w + b
output = tf.nn.softmax(tf.matmul(layers3, w4) + b4) 



# activate
'''
# hypothesis = x * w + b
layers1 = tf.nn.relu(tf.matmul(x, w) + b)    # relu
layers2 = tf.nn.elu(tf.matmul(x, w) + b)     # elu
layers3 = tf.nn.selu(tf.matmul(x, w) + b)    # selu
layers4 = tf.sigmoid(tf.matmul(x, w) + b)    # sigmoid
layers = tf.nn.dropout(layers4, keep_prob=0.3)    # Dropout
'''


# cost = tf.reduce_mean(tf.square(hypothesis-y))      # mse
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy :  y 값이 0과 1 사이로 바꼈으니까
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))         # categorical_crossentropy
cost = tf.losses.softmax_cross_entropy(y, output)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())


# 3. 훈련
for epochs in range(120):
    cost_val, hy_val, _ = sess.run([cost, output, train],
                        feed_dict={x:x_train, y:y_train})
    if epochs % 30 == 0:        # 10번에 1번씩 출력
        print(epochs, "cost :", cost_val, "\n", hy_val)


# 4. 평가, 예측
predicted = tf.cast(output > 0.5, dtype=tf.float32) # 0.5보다 크면 1, 작으면 0      # cast : 조건에 부합하면 1, 부합하지 않으면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))    # tf.equal : predicted, y이 동일하면 1, 아니면 0    # reduce_mean : 평균

c, a = sess.run([predicted, accuracy], feed_dict={x:x_test, y:y_test})

print("========================================================")
print("예측값 :\n", hy_val,
      "\n 예측결과값 :\n", c, "\n Accruacy :", a)

sess.close()

'''
 Accruacy : 0.97486
'''
