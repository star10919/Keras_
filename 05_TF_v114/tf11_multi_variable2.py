import tensorflow as tf
tf.compat.v1.set_random_seed(66)

### 행렬곱 : matmul 사용

x_data = [[73, 51, 65],         # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33,100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]    # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([3,1]), name='weight')       # x_data(5, 3) * w(3, 1) = hypothesis(5, 1)
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')                     # 3      3 동일해야 함 
'''
### 행렬의 곱
(a, b) * (c, d)
  => b랑 c의 크기가 동일해야 곱할 수 있음
결과값 shape : (a, d)
'''

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b       # x 먼저 넣어야 함!!!!!!!!!!!!!!!!!!!!!!!!(w보다)
print("잘나온다")






cost = tf.reduce_mean(tf.square(hypothesis-y))      # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:        # 10번에 1번씩 출력
        print(epochs, "cost :", cost_val, "\n", hy_val)

sess.close()