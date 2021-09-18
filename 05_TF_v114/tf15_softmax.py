from re import X
import numpy as np
import tensorflow as tf

tf.set_random_seed(66)


# 1. 데이터
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 2, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],     # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],     # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],     # 0
          [1, 0, 0]]
# print(x_data.shape, y_data.shape)   # (8, 4), (8, 3)



# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])   # input
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])   # output

w = tf.compat.v1.Variable(tf.random.normal([4, 3]), name='weight')       # [x열, y열]
b = tf.compat.v1.Variable(tf.random.normal([3]), name='bias')       # [3] = [1, 3]

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)     # softmax : 0과 1사이 (나올 3개의 값의 합은 1)


# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))       

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
# train = optimizer.minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)  # 위랑 동일



# 3. 훈련
# sess = tf.Session()       # 세션 열어주기
with tf.Session() as sess:      # with 문 쓰면 sess.close() 안 써도 됨.(with끝나면 세션 종료되니까)
    sess.run(tf.global_variables_initializer())     # 변수 초기화

    for epochs in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if epochs % 200 == 0:        # 10번에 1번씩 출력
            print(epochs, "cost :", cost_val)



# 4. 예측
    results = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print(results, sess.run(tf.argmax(results, 1)))

# sess.close()