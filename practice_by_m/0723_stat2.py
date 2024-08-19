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


import numpy as np 

np.arange(3).sum()/4

#기대값 16-

np.unique((np.arange(33) - 16) **2)
sum(np.unique((np.arange(33) - 16) **2) * (2/33))

np.unique((np.arange(x) - 16) **2)
sum(np.unique((np.arange(x - 16) **2) * (2/33))


(np.arange(33))


#E[X^2]
sum(x **2 * (1/33))

#Var(X) = E[X^2] 

x = np.arange(4)
x
pro_x = np.array([1/6,2/6,2/6,1/6])
pro_x

# 기대값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex **2

sum((x-Ex)**2 *pro_x)

----------------------------
x = np.arange(99)
x
x_1_50_1 = np.concatenate((np.arange(1, 51), np.arange(49, 0, -1)))/2500
pro_x = x_1_50_1/2500

Ex = sum(x * pro_x)
Exx = sum(x **2 * pro_x)

# 분산
Exx - Ex **2
sum((x-Ex)**2 *pro_x)

---------------------------------------
 x = 0,2.4.6
x = np.arange(4)*2
x
pro_x = np.array([1/6,2/6,2/6,1/6])
pro_x

# 기대값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)

# 분산
Exx - Ex **2

sum((x-Ex)**2 *pro_x)

---------------------------------------------------



