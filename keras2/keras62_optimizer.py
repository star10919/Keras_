import numpy as np

# learning_rate (커스터마이징)적용

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
# model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.0001)    # learning_rate (커스터마이징)적용   /   learning_rate를 줄이면 epochs는 그만큼 늘려줘야 된다.(다 돌게 하려면)
# optimizer = Adagrad(lr=0.0001)
# optimizer = Adadelta(lr=0.0001)
# optimizer = Adamax(lr=0.0001)
# optimizer = RMSprop(lr=0.0001)
# optimizer = SGD(lr=0.0001)
optimizer = Nadam(lr=0.0001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss :', loss, '결과물 :', y_pred)

# learing_rate 전
# loss : 7.140954629914278e-14 결과물 : [[10.999999]]



# learing_rate 후
# optimizer = Adam(lr=0.01)
# loss : 2.557953780973725e-14 결과물 : [[10.999999]]

# optimizer = Adam(lr=0.001)
# loss : 4.799780128905695e-09 결과물 : [[11.000141]]

# optimizer = Adam(lr=0.0001)
# loss : 7.356792934842815e-07 결과물 : [[11.001414]]



# optimizer = Adagrad(lr=0.01)
# loss : 8.003824768820778e-05 결과물 : [[11.007174]]

# optimizer = Adagrad(lr=0.001)
# loss : 6.4525256675551645e-06 결과물 : [[11.003543]]

# optimizer = Adagrad(lr=0.0001)
# loss : 0.0005785648827441037 결과물 : [[10.96945]]



# optimizer = Adadelta(lr=0.01)
# loss : 3.001180084538646e-05 결과물 : [[11.005594]]

# optimizer = Adadelta(lr=0.001)
# loss : 0.00047688354970887303 결과물 : [[10.987426]]

# optimizer = Adadelta(lr=0.0001)
# loss : 22.361583709716797 결과물 : [[2.6085021]]



# optimizer = Adamax(lr=0.01)
# loss : 0.07183406502008438 결과물 : [[11.387467]]

# optimizer = Adamax(lr=0.001)
# loss : 2.8975464374525473e-06 결과물 : [[10.997632]]

# optimizer = Adamax(lr=0.0001)
# loss : 7.857159653212875e-05 결과물 : [[10.988484]]



# optimizer = RMSprop(lr=0.01)
# loss : 0.7953788042068481 결과물 : [[9.085996]]

# optimizer = RMSprop(lr=0.001)
# loss : 2.0031020641326904 결과물 : [[8.338935]]

# optimizer = RMSprop(lr=0.0001)
# loss : 0.025990214198827744 결과물 : [[11.2794695]]



# optimizer = SGD(lr=0.01)
# loss : nan 결과물 : [[nan]]

# optimizer = SGD(lr=0.001)
# loss : 7.75915032136254e-06 결과물 : [[10.994045]]

# optimizer = SGD(lr=0.0001)
# loss : 0.0006133695133030415 결과물 : [[10.96741]]



# optimizer = Nadam(lr=0.01)
# loss : 0.0 결과물 : [[11.]]

# optimizer = Nadam(lr=0.001)
# loss : 6.266986712563649e-13 결과물 : [[10.999998]]

# optimizer = Nadam(lr=0.0001)
# loss : 6.405725343938684e-06 결과물 : [[10.995515]]