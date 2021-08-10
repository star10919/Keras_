from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### xor gate문제 단층 퍼셉트론으로 해결

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = LinearSVC()
# model = SVC()   # SVC는 LinearSVC보다 향상됨(다층 포함됨)
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))  # 단층 퍼셉트론(LinearSVC와 유사)

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 :", y_predict)

y_predict = np.round(y_predict, 0)      # 반올림해서 0 or 1 의 값으로 도출시키기 위함
print(y_predict)

acc = accuracy_score(y_data, y_predict)
print('acc_score : ', acc)

results = model.evaluate(x_data, y_data)
print('model.score :', results[1])


# acc = accuracy_score(y_data, y_predict)       #딥러닝이라 score 안먹힘 / evaluate 써야 함
# print("accuracy_score :", acc)


'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 : [[0.47034812]
 [0.6845783 ]
 [0.59896845]
 [0.78496   ]]
[[0.]
 [1.]
 [1.]
 [1.]]
acc_score :  0.75
1/1 [==============================] - 0s 88ms/step - loss: 0.7660 - acc: 0.7500
model.score : 0.75
'''