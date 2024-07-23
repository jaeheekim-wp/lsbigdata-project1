import numpy as np 
import matplotlib.pyplot as plt

data = np.random.rand(10)
data

#히스토그램 - 빈도표 
#bins데이터를 나누는 구간의 수나 경계 정의
#alpha 히스토그램의 투명도 

#sum(data < 1.8)

plt.hist(data,bins=4, alpha= 0.7, color="blue")
plt.title("Histogram of Numpy Vector")
plt.xlabel("value")
plt.ylabel("frequency")
plt.grid(True)
plt.show()
plt.clf()

#종 모양 히스토그램 
#01
data = np.random.rand(10000,5).mean(axis =1)
data
plt.hist(data, bins=30, alpha= 0.7, color="blue")
plt.title("Histogram of Numpy Vector")
plt.xlabel("value")
plt.ylabel("frequency")
plt.grid(True)
plt.show()
plt.clf()

#02
np.random.rand(50000).reshape(-1,5).mean(axis=1)

