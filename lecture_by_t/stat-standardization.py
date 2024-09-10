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
# 이런 표준화 과정을 통해 다른 시험이나 평가에서 나온 점수들을 '같은 기준'으로 비교.


# X ~ N(3, 7^2)
# Z = X - mu / sigma
# x = mu + Z * sigma
from scipy.stats import norm
# 하위 25 %
x=norm.ppf(0.25, loc=3, scale=7)
x
z=norm.ppf(0.25, loc=0, scale=1)
z

x
3 + z * 7
# 표준화를 진행해서 동일함 

norm.cdf(5, loc=3, scale=7)
norm.cdf(2/7, loc=0, scale=1)

norm.ppf(0.975, loc=0, scale=1) #1.96

# 표준 정규분포에서 표본 1000개 뽑고  pdf 그리기 
z=norm.rvs(loc=0, scale=1, size=1000)
z

norm.ppf(0.975, loc = 0, scale = 1)


# 표준 정규분포에서 표본 1000개 뽑고  pdf 그리기 
x = norm.rvs(loc=0, scale=1, size=1000)
sns.histplot(x, stat="density") # 스케일 맞춰줌
plt.show()

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

# -------------------------------------------

# 강사님 코드 풀이 

x = z * np.sqrt(2) + 3
sns.histplot(z, stat="density", color="grey")
sns.histplot(x, stat="density", color="green")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2) # 표준 정규분포 
plt.plot(z_values, pdf_values2, color='blue', linewidth=2)

plt.show()
plt.clf()

# 그니까 우리는 x의 범위가 필요하고, pdf 값이 필요해
# 그거를 plot 그래프로 x자리에 x의 범위 구한거, y자리에 pdf값 구한거

# -------------------------------------------
# sample
# X~N(5, 3^)
x = norm.rvs(loc=5, scale=3, size=1000)

# 표준화 
z=(x - 5)/3
sns.histplot(z, stat="density", color="grey")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

# -------------------------------------------

# t - 검정 통계량 ( 기초통계 74p)
# 표본 평균의 분포 x_bar = N (mu^, sigma^/n)  mu^ 예측치 ,즉 모분산 정보 없음 
# 모분산 정보가 없으므로 가장 가까운 값인 표본 표준편차로 대체한다.
# 표본표준편차로 나눠도 여전히 정규분포를 따를까? 


# sample2 
# X~N(5, 3^)  평균 5 /분산 9 (3^) / 표준편차 3 
# 1.x표본 10개 뽑아서 표본 분산값 계산 
# 2.x표본 1000개 뽑음 
# 3.계산한 분산s^ 값으로 sigma^(모분산) 대체한 표준화를 진행
# 4.Z의 히스토그램 그리기 == 표준정규분포 pdf 


#1.
x=norm.rvs(loc=5, scale=3, size=20)
s=np.std(x, ddof=1)
s
# s_2 = np.var(x, ddof=1) 표본 분산 

#2.
x=norm.rvs(loc=5, scale=3, size=1000)

# 표준화
z=(x - 5)/s
# z=(x - 5)/3
sns.histplot(z, stat="density", color="grey")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()
# -------------------------------------------

# [t 분포] X ~ t(n)
## 모수가 1개인 분포
## t 분포 특징: 종 모양, 대칭 분포, 중심이 0
## 모수 n: '자유도'라고 부르고, 퍼짐을 나타냄 # df=degree of freedom
## 따라서, n이 작으면 분산이 커짐
## t.OOO(k, df(n)) - OOO은 pdf/cdf/ppf/rvs

# 표본의 크기가 작은 경우에 평균을 비교하거나 
# 평균에 대한 신뢰구간을 계산할 때 유용


from scipy.stats import t

# t.pdf
# t.ppf
# t.cdf
# t.rvs
# 자유도가 4인 t분포의 pdf를 그려보세요!

t_values = np.linspace(-4, 4, 100)
pdf_values = t.pdf(t_values, df=4) 
# df = n 자유도 ( 숫자 바꿔서 체크해보기) > 커질수록 표준 분포랑 비슷해짐 
plt.plot(t_values, pdf_values, color='red', linewidth=2)

# 표준정규분포 겹치기
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='black', linewidth=2)
plt.show()
plt.clf()

# 정규 분포와의 차이: 
# T 분포는 정규 분포와 비슷하지만, 꼬리가 더 두껍습니다. 
# 이는 데이터가 적을 때 극단적인 값이 나올 확률이 더 높다는 것을 의미
# 데이터가 많아질수록 T 분포는 정규 분포와 더 유사해집니다.

# -------------------------------------------

# X ~ ?(mu, sigma^2)
# X bar ~ N(mu, sigma^2/n)
# X bar ~= t(x_bar, s^2/n) 자유도(df)가 n-1인 t 분포

x=norm.rvs(loc=15, scale=3, size=16, random_state=42)
x
x_bar=x.mean()
n=len(x)

# 모분산을 모를 때, 표본 분산으로 쓸수 있다 
# 표본 분산 # np.var(x, ddof = 1) 

# 모분산을 모를때: 모평균에 대한 95% 신뢰구간을 구해보자! 
# t분포 활용 
# 신뢰구간 계산: 평균 ± (z-값 * SE)
# SE = std/ sqrt(n)
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)

# 모분산(3^2)을 알때: 모평균에 대한 95% 신뢰구간을 구해보자!
# 표준 정규분포 활용 
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)



# 정리 
# t-검정 통계량의 분포 
 ## 표본의 크기가 크면 (100개 이상) 거의 표준 정규분포를 따른다고 봐도 무방 
 ## 하지만 표본의 크기가 작은 경우, 자유도가 n-1인 t 분포를 따른다.

# -------------------------------------------
# 표본표준편차로 데이터를 나누는 것만으로는 데이터를 표준정규분포로 만들 수 없어요.
# 데이터를 표준정규분포로 만들기 위해서는 표준화라는 과정을 거쳐야 함.
# 이 과정을 거치면, 데이터는 평균이 0이고 표준편차가 1인 분포를 따르게 됩니다.
# 그러나 이렇게 표준화된 데이터가 꼭 표준정규분포 (즉, 정규분포와 동일한 형태의 분포) 를 
# 따르지는 않을 수 있습니다. 표준화는 데이터를 정규분포와 같은 형태로 만들기 위한 조건이 아니며, 
# 원래 데이터가 정규분포를 따른다는 전제가 있어야 결과도 정규분포의 형태를 갖추게 됩니다.

# 즉,
# 표본표준편차로 나눈다고 해서 무조건 표준정규분포가 되는 것은 아니지만, 
# 원래 데이터가 정규분포를 따른다면, 표준화 과정을 통해 표준정규분포로 변환할 수 있습니다.

