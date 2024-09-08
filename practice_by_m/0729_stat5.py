import pandas as pd
import numpy as np 
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt




# y = 2x 그리기
# 점을 직선으로 이어서 표현

x = np.linspace(0, 8, 2)
y = 2 * x
plt.plot(x, y, color="black")

# 포인트로 표현
# plt.scatter(x, y, s = 2)
plt.show()
plt.clf()


# y = 2^x 그리기
x = np.linspace(-8, 8, 100)
y = x ** 2
plt.plot(x, y, color="black")
#plt.scatter(x, y, s = 3)

plt.xlim(-10,10)
plt.ylim(0,40)
plt.gca().set_aspect('equal' , adjustable = 'box')

# plt.axis('equal')는 xlim/ylim 이랑 같이 사용 x
plt.show()
plt.clf()



# p.57 신뢰구간 구하기
# 선생님 
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean() # 기댓값이라고 하면 안댐 
len(x) # n =16
6 # 표준편차(sigma)
0.1 # 알파 /// 1-알파 :0.9 (신뢰수준)
Z 0.05 

z_005 = norm.ppf(0.95, loc = 0, scale= 1)
z_005

# 신뢰구간 
x.mean() + z_005 * 6 / np.sqrt(16)
x.mean() - z_005 * 6 / np.sqrt(16)

# 데이터로부터 E[X^] 구하기
# X ~ N(3,5^)
x = norm.rvs(loc = 3, scale = 5, size = 10000)
x ** 2
np.mean(x ** 2) # 기댓값 
# sum(x**2) / (len(x) - 1)

np.mean((x - x**2) / (2*x))

# 몬테카를로 적분
# 확률변수 기대값을 구할때, 표본을 많이 뽑은 후 원하는 형태로 변형,
# 평균을 계산해서 기대값을 구하는 방법 

# ---------------------------------------------------------------------

# X~N(3,5^) 
# sigma^ = 25  = var(x)
# 표본 10만개 추출
# 표본분산 s^2을 구해보세요 

np.random.seed(20240729)
x = norm.rvs(loc = 3, scale = 5, size = 100000)

# 표본평균 (기댓값mu을 추정할 때 쓰고 값은 거의 비슷하다.)
x_bar = x.mean()

# 표본분산s^2 코드화 수식 
s_2 = sum((x - x_bar)**2) / (100000-1)
s_2

# 표본 분산 
# np.var로 사용하면 됨. 

# np.var(x) 사용하면 안됨 주의! 
# np.var(x, ddof = 0) # n으로 나눈 값 

np.var(x, ddof = 1) # n-1로 나눈 값 (표본 분산)

# n-1 vs. n
x = norm.rvs(loc = 3, scale = 5, size = 20)
np.var(x)
np.var(x, ddof = 1)







