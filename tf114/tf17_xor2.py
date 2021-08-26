import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

### 인공지능의 겨울을 극복하자(다층퍼셉트론(mlp)으로)
# perceptron -> mlp


# 1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)       # (4, 2)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)                    # (4, 1)






# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 히든레이어 1
w1 = tf.compat.v1.Variable(tf.random.normal([2,11]), name='weight1')       # [윗 레이어의 열, 내가 주고 싶은 노드의 개수]
b1 = tf.compat.v1.Variable(tf.random.normal([11]), name='bias1')                     

# hypothesis = x * w + b
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)


# 히든레이어 2
w2 = tf.compat.v1.Variable(tf.random.normal([11,5]), name='weight2')       # [윗 레이어의 열, 내가 주고 싶은 노드의 개수]
b2 = tf.compat.v1.Variable(tf.random.normal([5]), name='bias2')                     

# hypothesis = x * w + b
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)     # 전레이어와 가중치를 곱하는 것임


# 아웃풋레이어
w3 = tf.compat.v1.Variable(tf.random.normal([5,1]), name='weight3')      # 아웃풋 열은 최종 열로 맞춰줘야 함
b3 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias3')  

# hypothesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3) 






# cost = tf.reduce_mean(tf.square(hypothesis-y))      # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy :  y 값이 0과 1 사이로 바꼈으니까

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())


# 3. 훈련
for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:        # 10번에 1번씩 출력
        print(epochs, "cost :", cost_val, "\n", hy_val)


# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 0.5보다 크면 1, 작으면 0      # cast : 조건에 부합하면 1, 부합하지 않으면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))    # tf.equal : predicted, y이 동일하면 1, 아니면 0    # reduce_mean : 평균

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})

print("========================================================")
print("예측값 :\n", hy_val,
      "\n 예측결과값 :\n", c, "\n Accruacy :", a)

sess.close()

'''
예측값 :
 [[0.13122827]
 [0.8483398 ]
 [0.87046677]
 [0.17205693]]
 예측결과값 :
 [[0.]
 [1.]
 [1.]
 [0.]]
 Accruacy : 1.0
'''