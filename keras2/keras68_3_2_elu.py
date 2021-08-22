import numpy as np
import matplotlib.pyplot as plt

# elu : 0보다 작은 경우는 alpha값을 이용해서 그래프를 부드럽게 만든다.

def elu(x, alpha):
    return (x>0)*x + (x<=0)*(alpha*(np.exp(x)-1))

x = np.arange(-5, 5, 0.1)
y = elu(x, 2)

plt.plot(x, y)
plt.grid()
plt.show()