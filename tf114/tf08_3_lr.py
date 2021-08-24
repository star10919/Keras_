# 실습
# tf08_2 파일의 lr을 수정해서
# epoch가 2000번이 아니라 100번 이하로 줄여라
# 결과치는 step=100 이하, w=1.9999, b=0.9999

### y = wx + b
# w, b : 변수
# x, y : placeholder

### 실습
# predict 하는 코드 추가! & x_test라는 placeholder생성
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]


# wx + b  ->  hypothesis  ->  loss  ->  train

import tensorflow as tf
tf.set_random_seed(77)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable([1], dtype = tf.float32) # 랜덤하게 내 마음대로 넣어준 초기값
# b = tf.Variable([1], dtype = tf.float32)

W = tf.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32) # 랜덤하게 내 마음대로 넣어준 초기값
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32)



hypothesis = x_train * W + b    # 모델 구현
# f(x) = wx + b



loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

optimizer = tf.train.AdamOptimizer(learning_rate=0.821)
# optimizer = tf.train.AdagradOptimizer(learning_rate=6)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.429998)

train = optimizer.minimize(loss)

sess = tf.Session()        # 방법1
# with tf.Session() as sess:   # 방법2

sess.run(tf.global_variables_initializer()) # 변수초기화(반드시 해줘야 함!!)

for step in range(2001):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                            feed_dict={x_train:[1,2,3], y_train:[3,5,7]})     #  _, loss_val : train값은 보지 않고, loss의 값만 보겠다.
    if step % 20 ==0:       # 20번마다 1번씩 출력
        # print(step, sess.run(loss), sess.run(W), sess.run(b))       # 모델 돌때마다 변수들은 갱신됨
        print(step, loss_val, W_val, b_val)



x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

hypothesis_2 = x_test * W_val + b_val

pred_1 = sess.run(hypothesis_2, feed_dict={x_test:[4]})
pred_2 = sess.run(hypothesis_2, feed_dict={x_test:[5, 6]})
pred_3 = sess.run(hypothesis_2, feed_dict={x_test:[6, 7, 8]})
print(pred_1)
print(pred_2)
print(pred_3)


