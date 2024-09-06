import numpy as np

x = np.arange(2, 13)
prob = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36

Ex =np.sum(x * prob)  # 7
# E[(X-Ex)^2]
Var = np.sum((x - Ex)**2 * prob) # 5.83


# 2X+3
2 * Ex + 3 # 17
np.sqrt(4 * Var) # 4.83


