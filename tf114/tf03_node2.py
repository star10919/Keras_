# 실습
# 덧셈, 뺄셈, 곱셈, 나눗셈

import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

node3 = tf.add(node1, node2)    # 덧셈
node4 = tf.subtract(node1, node2)    # 뺄셈
node5 = tf.multiply(node1, node2)    # 곱셈
node6 = tf.div(node1, node2)    # 나눗셈

sess = tf.Session()
print("덧셈 :", sess.run(node3))
print("뺄셈 :", sess.run(node4))
print("곱셈 :", sess.run(node5))
print("나눗셈 :", sess.run(node6))

'''
덧셈 : 5.0
뺄셈 : -1.0
곱셈 : 6.0
나눗셈 : 0.6666667
'''