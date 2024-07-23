# numpy, pandas, matplotlib 설치해볼것!

import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(10)
# sum(data < 0.18)

# 히스토그램 그리기
plt.clf()
plt.hist(data, bins=4, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()


# x=np.random.rand(50000) \
#     .reshape(-1, 5) \
#     .mean(axis=1)
x=np.random.rand(10000, 5).mean(axis=1)
plt.hist(x, bins=30, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()
