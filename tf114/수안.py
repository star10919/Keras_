from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.metrics import r2_score

tf.set_random_seed(66)

datasets = load_boston()
x_data = datasets.data
y_data = datasets.target

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.compat.v1.random_normal([13, 1]), name='weight') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost) # 가장 작은 loss를 구한다.

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(10):
    _, cost_val, hy_val = sess.run([train, cost, hypothesis], 
              feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "\n", hy_val)
hy_val = y_data.reshape(hy_val.shape[0],)
print("스코어 : ",r2_score(hy_val, y_data))
print(y_data.shape)
print(hy_val.shape)
#