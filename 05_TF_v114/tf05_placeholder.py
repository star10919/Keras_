import tensorflow as tf

### placeholder : 빈공간
# placeholer & feed_dict 세트임!!(빈공간(placeholer)에 값(feed_dict) 넣어줘야 하니까)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()





a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))     # feed_dict : 안에다 값을 넣어줌(딕셔너리 형태로)
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2}))
