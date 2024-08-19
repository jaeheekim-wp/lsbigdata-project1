import pandas as pd
import numpy as np

old_seat=np.arange(1, 29)


np.random.seed(20240729)
# 1~28 숫자 중에서 중복 없이 28개 숫자를 뽑는 방법
new_seat=np.random.choice(old_seat, 28, replace=False)

result=pd.DataFrame(
    {"old_seat": old_seat,
     "new_seat": new_seat}
)

result.to_csv(result, "result.csv")

--------------------------------------------------------------

# y=2x 그래프 그리기
# 점을 직선으로 이어서 표현
import matplotlib.pyplot as plt

x = np.linspace(0, 8, 2)
y = 2 * x
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.show()
plt.clf()

# y = x^2 를 점 3개 사용해서 그리기
x = np.linspace(-8, 8, 100)
y = x**2
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")

# x, y 축 범위 설정
plt.xlim(-10, 10)
plt.ylim(0, 40)
# 비율 맞추기
# plt.axis('equal')는 xlim, ylim과 같이 사용 x
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
plt.clf()

from scipy.stats import norm
import numpy as np
# 다음은 한 고등학교의 3학년 학생들 중 16명을 무작위로 선별하여 몸무게를 측정한 데이터이다. 
# 이 데이터를 이용하여 해당 고등학교 3학년 전체 남학생들의 몸무게 평균을 예측하고자 한다.
# 79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8
# 단, 해당 고등학교 3학년 남학생들의 몸무게 분포는 정규분포를 따른다고 가정한다.
# 1) 모평균에 대한 95% 신뢰구간을 구하세요.
# 2) 작년 남학생 3학년 전체 분포의 표준편차는 6kg 이었다고 합니다. 
# 이 정보를 이번 년도 남학생 분포의 표준편차로 대체하여 모평균에 대한 90% 신뢰구간을 구하세요.

x=np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])

# 표본 평균
# 기댓값이라고 하면 안댐 
x.mean()

# 표본 크기 n
len(x)

# 표준편차(sigma)
6 

# alpha (유의수준)
0.1 #  1-alpha :0.9 (신뢰수준)

Z 0.05 
z_005=norm.ppf(0.95, loc=0, scale=1) 

# 90% 신뢰구간
x.mean() + z_005 * 6 / np.sqrt(16)
x.mean() - z_005 * 6 / np.sqrt(16)
-------------------------------------------

# 데이터로부터 E[X^2] 구하기
x=norm.rvs(loc=3, scale=5, size=100000)

np.mean(x**2)
# sum(x**2) / (len(x) - 1)
np.mean((x - x**2) / (2*x))


# 몬테카를로 적분
# 확률변수 기대값을 구할때, 표본을 많이 뽑은 후 원하는 형태로 변형,
# 평균을 계산해서 기대값을 구하는 방법 

# ------------------------
# X~N(3,5^) 
# sigma^ = 25  = var(x)
# 표본 10만개 추출
# 표본분산 s^2을 구해보세요 

np.random.seed(20240729)
x=norm.rvs(loc=3, scale=5, size=100000)

# 표본평균 
# 기댓값을 추정할 때 쓰고 값은 거의 비슷하다.
x_bar = x.mean()

# 표본분산s^2 코드화 수식 
# s^2 = 표본 분산 / sigma^2 = 모분산 
s_2 = sum((x - x_bar)**2) / (100000-1)
s_2


# 표본 분산 
# np.var로 사용하면 됨. 

# np.var(x) 사용하면 안됨 주의! 
# np.var(x, ddof = 0) # n으로 나눈 값 

np.var(x, ddof=1) # n-1로 나눈 값 


# n-1 vs. n
x=norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x, ddof=1)
  

