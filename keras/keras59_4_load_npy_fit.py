import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1-1. 데이터 넘파이로드
# np.save('./_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])

x_train = np.load('./_save/_npy/k59_3_train_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일(ES), 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=50, batch_size=50,
                           validation_split=0.2)

results = model.evaluate(x_test, y_test)
print('binary :', results[0])
print('acc :', results[1])
