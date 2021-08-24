import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)        # constant : 상수(변하지 않는 고정 값)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node3)    # Tensor("Add:0", shape=(), dtype=float32)


### Session, run 통과시켜야 값이 출력됨!!!
sess = tf.Session()
print('node1, node2 :', sess.run([node1, node2]))       # node1, node2 : [3.0, 4.0]
print('sess.run(node3) :', sess.run(node3))             # sess.run(node3) : 7.0