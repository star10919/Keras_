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

results = model.evaluate(x_data, y_data)
print('model.score :', results[1])

# acc = accuracy_score(y_data, y_predict)
# print("accuracy_score :", acc)

'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 : [0 0 0 1]
model.score : 1.0
accuracy_score : 1.0
'''