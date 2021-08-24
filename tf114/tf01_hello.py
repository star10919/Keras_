import tensorflow as tf
print(tf.__version__)

# print('hello world')

hello = tf.constant("hello world")      # constant : 상수
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()     # tensorflow1 은 출력할 때,   반드시 Session을 선언하고
print(sess.run(hello))      # run 안에 넣어야 출력할 수 있음
# b'hello world'