import numpy as np
import matplotlib.pyplot as plt

### tanh : -1 ~ 1 사이

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()