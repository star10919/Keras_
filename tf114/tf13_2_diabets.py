# 실습
# diabet shape에 맞춰서, 최종결론값은 sklearn의 r2_score로 할 것
# pip install sklearn

from sklearn.datasets import load_diabetes
import tensorflow as tf
tf.set_random_seed(66)

datasets = load_diabetes()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

