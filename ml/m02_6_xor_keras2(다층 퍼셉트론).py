from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### xor gate문제 다층 퍼셉트론으로 해결

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = LinearSVC()
# model = SVC()   # SVC는 LinearSVC보다 향상됨(다층 포함됨)
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))  # 다층 퍼셉트론(SVC와 유사)
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='sigmoid'))



# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 :\n", y_predict)

y_predict = np.round(y_predict, 0)
print(y_predict)

acc = accuracy_score(y_data, y_predict)
print('acc_score : ', acc)

result = model.evaluate(x_data, y_data)
print('model_score : ', result[1])


# acc = accuracy_score(y_data, y_predict)       #딥러닝이라 score 안먹힘 / evaluate 써야 함
# print("accuracy_score :", acc)

'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :
 [[0.10601883]
 [0.95537454]
 [0.96742076]
 [0.02704337]]
[[0.]
 [1.]
 [1.]
 [0.]]
acc_score :  1.0
1/1 [==============================] - 0s 114ms/step - loss: 0.0546 - acc: 1.0000
model_score :  1.0
'''