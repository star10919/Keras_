import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])


# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[0.9531883 , 0.01933467, 0.93905675]], dtype=float32)>,
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=array([[ 0.9944706 , -0.37384486], [ 0.3210615 , -0.2729177 ], [-0.5919616 , -0.7304392 ]], dtype=float32)>,
<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=array([[-0.95255923],[ 0.6022178 ]], dtype=float32)>,
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]  
'''

print('====================================================================================')
print(model.trainable_weights)
'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.657431 ,  1.0375811,  0.7274983]], dtype=float32)>,
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=array([[ 0.38519132, -0.23749638],[-0.43617678,  0.8414396 ],[-0.13436306, -0.13127202]], dtype=float32)>,
<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=array([[ 0.33839738],[-1.2277272 ]], dtype=float32)>,
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

print(len(model.weights), len(model.trainable_weights))     # 6, 6  =3(w+b)=layer(w+b)