import numpy as np
import matplotlib.pyplot as plt

# sigmoid : 0 ~ 1 사이

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)   # -5에서 5까지 0.1씩 차이남
print(x)

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()
