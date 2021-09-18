import numpy as np
import matplotlib.pyplot as plt

# Leaky_Relu : Relu는 0보다 작은 입력신호에 대해 출력을 꺼버린다. / relu에 비해 연산의 복잡성이 크다는 것

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

x = np.arange(-5, 5, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()