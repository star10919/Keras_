### tf.compat.v1.disable_eager_execution : tensorflow2에서 1 사용할 수 있게 해줌

import tensorflow as tf
print(tf.__version__)       # 1.14.0




print(tf.executing_eagerly())   # True

tf.compat.v1.disable_eager_execution()    # disable_eager_execution : 즉시실행모드

print(tf.executing_eagerly())   # False




hello = tf.constant("hello world")      # constant : 상수
print(hello)    # 텐서1에서는 그냥 출력하면 자료구조형이 나옴
# Tensor("Const:0", shape=(), dtype=string) 
#                       스칼라

# sess = tf.Session()     # tensorflow1 은 출력할 때,   반드시 Session을 선언하고
sess = tf.compat.v1.Session()
print(sess.run(hello))      # run 안에 넣어야 출력할 수 있음
# b'hello world'