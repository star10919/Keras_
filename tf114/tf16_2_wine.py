# 실습 accuracy넣기

from re import X
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

tf.set_random_seed(66)


# 1. 데이터
datasets = load_wine()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)   # (178, 13) (178,)
print(x_data, y_data)   # 0, ..., 1, ..., 2 ...

y_data = y_data.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_data)
y_data = one.transform(y_data).toarray()
print(y_data, y_data.shape)     # (178, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.68, random_state=77)


# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])   # input
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])   # output

w = tf.compat.v1.Variable(tf.random.normal([13, 3]), name='weight')       # [x열, y열]
b = tf.compat.v1.Variable(tf.random.normal([1, 3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)     # softmax : 0과 1사이 (나올 3개의 값의 합은 1)


# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))       

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
# train = optimizer.minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00013).minimize(loss)  # 위랑 동일



# 3. 훈련
# sess = tf.Session()       # 세션 열어주기
with tf.Session() as sess:      # with 문 쓰면 sess.close() 안 써도 됨.(with끝나면 세션 종료되니까)
    sess.run(tf.global_variables_initializer())     # 변수 초기화

    for epochs in range(6000):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if epochs % 1000 == 0:        # 10번에 1번씩 출력
            print(epochs, "cost :", cost_val)



# 4. 예측
    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_pred, y_test)    # reduce_mean : 평균
    print("acc :", accuracy)

# sess.close()

'''
acc : 0.2982456140350877
'''