import tensorflow as tf
sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32)#, name='test')

init = tf.global_variables_initializer()        # 변수사용하려면 변수초기화 반드시 해야함!!(안하면 안돌아감)  /  그래프에 들어가기에 적합함
sess.run(init)

print("프린트 x 나왔나 확인 :", sess.run(x))


