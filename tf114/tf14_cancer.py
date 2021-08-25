# 실습
# breast_cancer shape에 맞춰서, 최종결론값은 2진분류-accruacy로 할 것
# pip install sklearn

from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(66)

datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)     # (569, 30) (569,)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, ])


# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,])

w = tf.compat.v1.Variable(tf.random.normal([30,1]), name='weight')       # x_data(5, 3) * w(3, 1) = hypothesis(5, 1)
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')                     # 3      3 동일해야 함 
'''
### 행렬의 곱
(a, b) * (c, d)
  => b랑 c의 크기가 동일해야 곱할 수 있음
결과값 shape : (a, d)
'''

# hypothesis = x * w + b            # 1차원일 때
hypothesis = tf.matmul(x, w) + b    # 다차원일 때(행렬곱)
# hypothesis = tf.sigmoid(tf.matmul(x, w) + b)   # 2진분류     # activation함수 : 출력되는 값을 tf.sigmoid로 감쌈

cost = tf.reduce_mean(tf.square(hypothesis-y))      # mse
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy :  y 값이 0과 1 사이로 바꼈으니까

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())


# 3. 훈련
for epochs in range(2000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:        # 10번에 1번씩 출력
        print(epochs, "cost :", cost_val, "\n", "hy_val :", hy_val)

from sklearn.metrics import r2_score, accuracy_score
r2 = accuracy_score(hy_val, y_data)
print("r2 스코어 :", r2)
print(y_data.shape)
print(hy_val.shape)

sess.close()



# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 0.5보다 크면 1, 작으면 0      # cast : 조건에 부합하면 1, 부합하지 않으면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))    # tf.equal : predicted, y이 동일하면 1, 아니면 0


