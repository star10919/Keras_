import tensorflow as tf
tf.set_random_seed(66)

x_data = [[73, 51, 65],         # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33,100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]    # (5, 1)

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3,1]), name=['weight'])
b = tf.Variable(tf.random_normal([1]), name='bias')
