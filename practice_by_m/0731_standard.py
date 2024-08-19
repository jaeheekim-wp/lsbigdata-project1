from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 표준 정규분포를 통한 표준화
# 데이터를 비교 가능한 표준 형식으로 변환하는 방법

# 표준 정규분포 : 평균이 0이고 표준편차가 1인 특별한 정규분포
# 모든 정규분포는 데이터를 평균과 표준편차를 사용해 변환하면 이 표준 정규분포로 변환 가능.


# 표준화 과정
# 01.평균과 표준편차 구하기:
## 먼저, 데이터의 평균(평균적인 값)과 표준편차(데이터가 평균에서 얼마나 퍼져있는지 측정)를 계산.

# 02.표준화 공식: Z = X - mu / sigma
# z = 표준화된 값 
# x = 원래 데이터 값 
# mu = 데이터의 평균 
# sigma = 데이터의 표준편차 

# 표준화된 데이터 값(Z)
 ## 데이터의 단위에 상관없이 동일한 기준으로 변환된 값
 ## 이렇게 변환된 데이터는 평균이 0이고, 표준편차가 1인 분포를 따르게 됨 >> 이게 표준 정규분포
 
# EX)
# 어떤 학생들의 시험 점수가 있을 때, 
# 시험의 평균이 70점이고 표준편차가 10점이라면, 
# 한 학생의 점수가 85점일 때 표준화된 점수는: 
# 85 - 70 / 10 = 1.5
# 표준화된 점수는 1.5.> 평균보다 1.5 표준편차 만큼 높은 점수라는 것을 의미. 
# 이런 표준화 과정을 통해 다른 시험이나 평가에서 나온 점수들을 같은 기준으로 비교.

-------------------------------------------------------------------------------------

# 하위 25 %
# X ~ N(3,7^)
x = norm.ppf(0.25, loc = 3, scale = 7)
z = norm.ppf(0.25, loc = 0, scale = 1)

x 

x = 3 + z *7 # 표준화를 진행해서 동일함 

norm.cdf(5, loc = 3, scale = 7)
norm.cdf(2/7, loc = 0, scale = 1)

norm.ppf(0.975, loc = 0, scale = 1)


# 표준 정규분포에서 표본 1000개 뽑고  pdf 그리기 
x = norm.rvs(loc=0, scale=1, size=1000)
sns.histplot(x, stat="density") # 스케일 맞춰줌

xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100) # 일정한 간격으로 숫자 출력
pdf_values = norm.pdf(x_values, loc=0, scale=1)
plt.plot(x_values, pdf_values, color="red", linewidth="2")
plt.show()
# plt.clf()

# x~n(3, sigma 2^ )
z = norm.rvs(loc=3, scale=np.sqrt(2), size=1000)
sns.histplot(z, stat="density") # 스케일 맞춰줌

zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100) # 일정한 간격으로 숫자 출력
pdf_values = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color="green", linewidth="2")
plt.show()
plt.clf()

--------------------------------------------------------------------------------

#강사샘 풀이

z=norm.rvs(loc=0, scale=1, size=1000)
z

x=z*np.sqrt(2) + 3
sns.histplot(z, stat="density", color="grey")
sns.histplot(x, stat="density", color="green")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='blue', linewidth=2)

plt.show()
plt.clf()

--------------------------------------------------------------------------------
# sample
# X~N(5, 3^)
# 표준화 확인 
x=norm.rvs(loc=5, scale=3, size=1000)
x

# 표준화 
z= (x-5)/3
sns.histplot(z, stat="density", color="grey")X~
sns.histplot(x, stat="density", color="green")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='blue', linewidth=2)

plt.show()
plt.clf()

--------------------------------------------------------------------------------

# sample2 
# X~N(5, 3^)  평균 5 /분산 9 (3^)
# 1.x표본 10개 뽑아서 표본 분산값 계산 
# 2.x표본 1000개 뽑음 
# 3.계산한 분산s^ 값으로 sigma^(모분산) 대체한 표준화를 진행
# 4.Z의 히스토그램 그리기 == 표준정규분포 pdf 

# 1
x=norm.rvs(loc=5, scale=3, size=10)
s = np.std(x, ddof = 1) # 표준편차 
s**2
# s_2 = np.var(x,ddof = 1) : 한결언니 방식 

# 2
x=norm.rvs(loc=5, scale=3, size=1000)

# 3 표준화 진행 
z= (x-5)/s
# z = 
sns.histplot(z, stat="density", color="grey")
# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
# 표본 사이즈가 작을때 이론값과 맞지 않는 문제 

plt.show()
plt.clf()

# t 분포에 대해서 알아보자 
# x ~t(df)
# 종모양, 대칭분포, 중심 0 
# 모수 df : 자유도라고 부름 - 퍼짐을 나타내는 모수(자유도 = n)
# df 이 작으면 분산 커짐.
# df 이 무한대로 가면 표준 정규 분포 
from scipy.stats import t
# t.pdf
# t.ppf
# t.cdf
# t.rvs

# 그니까 우리는 x의 범위가 필요하고, pdf 값이 필요해
# 그거를 plot 그래프로 x자리에 x의 범위 구한거, y자리에 pdf값 구한거

# [t 분포] X ~ t(n)
## 모수가 1개인 분포
## t 분포 특징: 종 모양, 대칭 분포, 중심이 0
## 모수 n: '자유도'라고 부르고, 퍼짐을 나타냄
## 따라서, n이 작으면 분산이 커짐
## t.OOO(k, df(n)) - OOO은 pdf/cdf/ppf/rvs

# 자유도가 4인 t 분포의 pdf 그리기 
t_values = np.linspace(-4, 4, 500)
pdf_values = t.pdf(t_values, df = 2) 
# df = n 자유도 ( 숫자 바꿔서 체크해보기) > 커질수록 표준 분포랑 비슷해짐 
plt.plot(t_values, pdf_values, color='red', linewidth=2)

# 표준 정규분포 겹치기 
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='black', linewidth=2)

plt.show()
plt.clf()

# X ~ ? (mu, sigma^2)
# X bar ~ N (mu, sigma^2/n)
# X bar ~=t (x_bar s^2/n) 자유도가 n-1인 t 분포 

x = norm.rvs(loc=15, scale=3, size=16, random_state = 42)
x
x_bar= x.mean()
n = len(x)

# 모분산을 모를 때 : 모평균에 대한 95퍼 신뢰구간  
x_bar + t.ppf(0.975, df = n-1) * np.std(x, ddof = 1)/ np.sqrt(n)
x_bar - t.ppf(0.975, df = n-1) * np.std(x, ddof = 1)/ np.sqrt(n)

# 모분산3^2를 알 때 : 모평균에 대한 95퍼 신뢰구간  
x_bar + t.ppf(0.975, loc=0, scale = 1)  * 3 / np,sqrt(n)
x_bar - t.ppf(0.975, loc=0, scale = 1)  * 3 / np,sqrt(n)

