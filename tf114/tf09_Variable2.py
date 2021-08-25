import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b

# case1. Session -> .run
sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("aaa :", aaa) # aaa : [0.3]
sess.close()

# case2. InteractiveSession -> .eval
sess = tf.InteractiveSession()      # InteractiveSession : Session 이랑 똑같음(명칭만 다름)
sess.run(tf.global_variables_initializer())
bbb = W.eval()      # 변수.eval
print("bbb :", bbb) # bbb : [0.3]
sess.close()

# case3. Session -> .eval(session=)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc :", ccc) # ccc : [0.3]
sess.close()