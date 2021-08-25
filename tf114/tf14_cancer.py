# 실습
# breast_cancer shape에 맞춰서, 최종결론값은 2진분류-accruacy로 할 것
# pip install sklearn

from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(66)

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (569, 30) (569,)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])





# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 0.5보다 크면 1, 작으면 0      # cast : 조건에 부합하면 1, 부합하지 않으면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))    # tf.equal : predicted, y이 동일하면 1, 아니면 0


