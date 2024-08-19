from scipy.stats import uniform
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# uniform 28p
# 균일 분포
# loc은 구간시작점, scale은 구간 길이 ! 주의 
# loc = a / scale = b-a

# uniform.pdf(x, loc=0, scale=1)
# uniform.cdf(x, loc=0, scale=1)
# uniform.ppf(q, loc=0, scale=1)
# uniform.rvs(loc=0, scale=1, size=None, random_state=None)

# X~U(2~6)
uniform.rvs(loc=2, scale = 4, size =1)

k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color = "black")
plt.show()
plt.clf()

p(x<3.25 )= ?
uniform.cdf(3.25, loc=2, scale=4)

p(5<x<8.39) = ?
uniform.cdf(8.39, loc=2, scale=4) -  uniform.cdf(5, loc=2, scale=4) 
uniform.cdf(6, loc=2, scale=4) -  uniform.cdf(5, loc=2, scale=4) 
# 6까지의 균일 분포니까 동일값 나옴옴

상위 7%
uniform.ppf(0.93, loc=2, scale=4)

# 표본 20개 뽑고 표본평균 계산 
x = uniform.rvs(loc=2, scale = 4, size = 20 )
x.mean()

x = uniform.rvs(loc=2, scale = 4, size = 20, random_state = 42)
x.mean()

x = uniform.rvs(loc=2, scale = 4, size = 20*1000, random_state = 42)
x = x.reshape(1000,20)
x.shape
blue_x=x.mean(axis=1)
blue_x

plt.hist(blue_x)
sns.histplot(blue_x)
plt.show()

# 히스토그램

sns.histplot(blue_x, stat="density")
plt.show()
plt.clf()


# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.333333/20) # 왜 20이 나오지? 아 표본 20개
# 분산 
uniform.var(loc=2, scale=4) #1.3333
# 기댓값
uniform.expect(loc=2, scale=4)


# Plot the normal distribution PDF ( 그래프 얹기 )
xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()


------------------------------------------
# X~U(2~6)
# 신뢰구간 

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.333333/20)

# Plot the normal distribution PDF 
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.3333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()

norm.ppf(0.025,loc=4,scale=np.sqrt(1.3333/20))
norm.ppf(0.975,loc=4,scale=np.sqrt(1.3333/20))
            
# 표본 평균(파란벽돌) 점 찍기 
blue_x = uniform.rvs(loc=2, scale = 4, size = 20 ).mean()
# 95% 신뢰구간- 자료38p
# norm.ppf
a = blue_x + 1.96 * np.sqrt(1.3333/20)   # 왜 1.96 ???
b = blue_x - 1.96 * np.sqrt(1.3333/20)
plt.scatter(blue_x, 0.002,
            color='blue', zorder=10, s=10)
plt.axvline(x=a, color='blue', 
            linestyle='--', linewidth=2)
plt.axvline(x=b, color='blue', 
            linestyle='--', linewidth=2)
# 기댓값 
plt.axvline(x=4, color='green', 
            linestyle='--', linewidth=2)

plt.show()
plt.clf()


blue_x = uniform.rvs(loc=2, scale = 4, size = 20 ).mean()
x = uniform.rvs(loc=2, scale = 4, size = 20 )
x.mean()

x = uniform.rvs(loc=2, scale = 4, size = 20, random_state = 42)
x.mean()

x = uniform.rvs(loc=2, scale = 4, size = 20*1000, random_state = 42)
x = x.reshape(1000,20)
x.shape
blue_x=x.mean(axis=1)
blue_x

plt.hist(blue_x)
sns.histplot(blue_x)
plt.show()



sns.histplot(blue_x, stat="density")
plt.show()
plt.clf()





