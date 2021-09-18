import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))        # 총합은 1

x = np.arange(1, 5)
y = softmax(x)

ratio = y

plt.pie(ratio, labels=y, shadow=False, startangle=90)        # 파라미터 주석달아기
plt.show()