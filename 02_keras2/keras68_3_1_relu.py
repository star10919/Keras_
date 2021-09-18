import numpy as np
import matplotlib.pyplot as plt

# relu : min이 0

def relu(x):
    return np.maximum(0, x)     # 0보다 큰건 유지, 0보다 작은건 0

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()


# 과제
# elu, selu, reaky relu ...
# 68_3_2, 3, 4 ..로 만들 것!!