import tensorflow as tf

tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
print(W)
#  <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>


# case1. Session -> .run
sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("aaa :", aaa) # [2.2086694]
sess.close()

# case2. InteractiveSession -> .eval
sess = tf.InteractiveSession()      # InteractiveSession : Session 이랑 똑같음(명칭만 다름)
sess.run(tf.global_variables_initializer())
bbb = W.eval()      # 변수.eval
print("bbb :", bbb)
sess.close()

# case3. Session -> .eval(session=)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc :", ccc)
sess.close()