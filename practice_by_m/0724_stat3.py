from scipy.stats import bernoulli
from scipy.stats import binom

# 확률질량함수(pmf)
# P(X=1)
bernoulli.pmf(1, 0.3)
# P(X=0)
bernoulli.pmf(0, 0.3)

# 이항분포
# P(X = K | n, p)
# n 베르누이 확률변수 더한 갯수
# p : 1이 나올 확률

binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)


binom.pmf(2, n=2, p=0.3)

result = [binom.pmf(x, n=30, p=0.3) for x in range(31)]
result

binom.pmf(np.arange(31),n=30, p=0.3)

-----------------------------------------------------------
n = 54 /r = 26

 
import math 
math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54,26) 

1*2*3*4
import numpy as np 
# np.cumprod(np.array(1,5))[-1]
# fact_54 = np.cumprod(np.array(1, 55))[-1]
# ln

log(a * b) = log(a) + log(b)
log(1 * 2 * 3 * 4) =  log(1) + log(2) +  log(3) + log(4)


np.log(24)
sum(np.log(np.arange(1, 5)))
math.log((math.factorial(54))
sum(np.log(np.arange(1, 5)))

math.comb(2, 0) * 0.3**0 (1-0.3) ** 2
math.comb(2, 0) * 0.3**1 (1-0.3) **1
-----------------------------------------------------------------------
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# 이항 분포란?
# 이항 분포는 일련의 독립적인 시행에서 각 시행이 성공 또는 실패 
# 두 가지 결과만을 가지는 경우, 성공의 횟수를 나타내는 확률 분포.

# 이항 분포의 주요 파라미터:
# n: 시행의 총 횟수
# p: 각 시행에서 성공할 확률
# k: 성공한 횟수 


# binom.pmf 함수
# binom.pmf(k, n, p) 함수는 다음을 계산합니다:
# k: 성공의 횟수
# n: 시행의 총 횟수
# p: 각 시행에서 성공할 확률

# 쉽게 풀어 설명
# 생각해보세요:

# n = 2는 동전을 2번 던지는 것과 같고,
# p = 0.3는 동전이 앞면(1)이 나올 확률이 30%인 것.
# 이제 우리가 계산하는 것은 동전을 2번 던졌을 때 앞면이 몇 번 나오는지를 보는 것입니다.

# 앞면이 한 번도 안 나올 확률 (binom.pmf(0, 2, 0.3)):
# 동전을 2번 던졌을 때 2번 모두 뒷면이 나올 확률입니다.
# 확률은 약 49%입니다.

# 앞면이 한 번 나올 확률 (binom.pmf(1, 2, 0.3)):
# 동전을 2번 던졌을 때 1번은 앞면, 1번은 뒷면이 나올 확률입니다.
# 확률은 약 42%입니다.

# 앞면이 두 번 다 나올 확률 (binom.pmf(2, 2, 0.3)):
# 동전을 2번 던졌을 때 2번 모두 앞면이 나올 확률입니다.
# 확률은 약 9%입니다.

# 이항 분포는 일정한 시행 횟수에서 특정 횟수의 성공이 일어날 확률을 계산하는 데 매우 유용.



# X~B(n = 10,p = 0.36) 
# P(X=4)?
# binom.pmf(x, n, p)

binom.pmf(4, 10, 0.36)

# P(X<=4)?
binom.pmf([0,1,2,3,4], 10, 0.36).sum()
binom.pmf(np.arange(5), n = 10, p = 0.36).sum()

# 확률변수 x가 2보다 크고, 8보다 작거나 같을 확률을 구하시오.
binom.pmf(np.arange(3,9), n = 10, p = 0.36).sum()

# X~B(n = 30,p = 0.2) 
# 확률변수 x가 4보다 작거나, 25보다 크거나 같을 확률을 구하시오.
# P(X<4 or X>=25)?
a = binom.pmf(np.arange(4), n = 30, p = 0.2).sum()
b = binom.pmf(np.arange(25, 31), n = 30, p = 0.2).sum()
a+b

# 다른 방식 
# 4 확률의 총 합은 100, 즉 1이라는 개념 활용
# 1- P(4<=X<25)?
binom.pmf(p.arange(4, 25)

1 - binom.pmf(np.arange(4, 25), n = 30, p = 0.2).sum()


# rvs함수
# 표본 추출 함수 

# X1 ~ Bernulli( p = 0.3)
bernoulli.rvs(p = 0.3)                             # 1이 나올 확률 0.3
# X2 ~ Bernulli( p = 0.3)
bernoulli.rvs( p = 0.3) + bernoulli.rvs( p = 0.3)  # 즉 X1 +X2
binom.rvs(n = 2, p =0.3, size = 1 )          
# 동일 방식 binom 코드 활용
# n=2: 시행을 두 번 합니다. 예를 들어, 동전을 두 번 던지는 것과 비슷합니다.
# p=0.3: 각 시행에서 성공(동전의 앞면이 나오는 확률)이 30%입니다.
# size=1: 이러한 조건으로 1개의 난수를 생성합니다. (표본의 갯수) 
 ##한결언니 설명 - 이러한 동전 게임 을 몇 번 할건지 

binom.pmf(0, n = 2, p = 0.3) # 시행 두 번하고, 1의 확률이 30%일때, 0이 나올 확률
binom.pmf(1, n = 2, p = 0.3) # 시행 두 번하고, 1의 확률이 30%일때, 1이 나올 확률
binom.pmf(2, n = 2, p = 0.3) # 시행 두 번하고, 1의 확률이 30%일때, 2이 나올 확률

--------------------------------------------------------------------------------
# binom.rvs: 동전을 던졌을 때 무작위로 나올 결과를 시뮬레이션한다.
 ## random sample size 랜덤 샘플 함수 
# binom.pmf: 동전을 던졌을 때 특정 결과가 나올 확률을 계산한다.
 ## 확률 질량 함수 

# X ~ B(30, 0.26)
# 표본 30개를 뽑아보세요.
binom.rvs(n = 30, p =0.26, size = 30 )  

# X ~ B(30, 0.26)
# X의 기댓값은? # EX = np
30 * 0.26

------------------------------------------------------------------------------------
# X ~ B(30, 0.26) 시각화

# !pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
k = np.arange(31)
prob_x = binom.pmf(k, n=30, p =0.26)
prob_x
# prob_x
# [P(앞면이 0번 나올 확률), P(앞면이 1번 나올 확률), ..., P(앞면이 30번 나올 확률)]
# 을 계산합니다.

sns.barplot(prob_x)
plt.show()
plt.clf()

import pandas as pd
df = pd.DataFrame({"x" : x, "prob" : prob_x})
df


sns.barplot(data = df, x = "x", y = "prob")

-------------------------------------------------------------------------

# cdf : cumulatuve dist. funtion ㅌ_!
# 누적 확률 분포 함수
# F(X=x) = P(X<=X)

binom.cdf(4, n =30, p =0.26)

# 쉽게 풀어서 설명
# 상황 설정

# 당신은 동전을 30번 던집니다.
# 이 동전은 앞면이 나올 확률이 26%입니다.
# 우리는 동전을 30번 던졌을 때 앞면이 4번 이하로 나올 확률이 궁금합니다.
# binom.cdf(4, n=30, p=0.26)가 하는 일

# 이 함수는 동전을 30번 던졌을 때 앞면이 0번, 1번, 2번, 3번, 4번 나올 각각의 확률을 모두 더합니다.
# 즉, 앞면이 4번 이하로 나올 확률을 계산합니다.
# 예시
# binom.cdf(4, n=30, p=0.26)는 다음 확률들을 더한 값입니다:
# 동전을 30번 던졌을 때 앞면이 0번 나올 확률
# 동전을 30번 던졌을 때 앞면이 1번 나올 확률
# 동전을 30번 던졌을 때 앞면이 2번 나올 확률
# 동전을 30번 던졌을 때 앞면이 3번 나올 확률
# 동전을 30번 던졌을 때 앞면이 4번 나올 확률

binom.cdf(18, n =30, p =0.26) - binom.cdf(4, n =30, p =0.26)
binom.cdf(19, n =30, p =0.26) - binom.cdf(13, n =30, p =0.26)


x_1 = binom.rvs(n =30, p =0.26, size = 3)
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p =0.26)
prob_x
sns.barplot(prob_x, color = "blue")
# 포인트 추가하기 
plt.scatter(2, 0.002, color="red", zorder=100, s=10) #zorder는 레이어 개념
plt.axvline(x=7.8, color = "green", linestyle="--", linewidth=2)
plt.show()
plt.clf()


x_1 = binom.rvs(n =30, p =0.26, size = 3)
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p =0.26)
prob_x
sns.barplot(prob_x, color = "blue")
# 포인트 추가하기 
plt.scatter(x_1, np.repeat(0.002, 3), color="red", zorder=100, s=10) #zorder는 레이어 개념
# 기댓값 표현 
plt.axvline(x=7.8, color = "green", linestyle="--", linewidth=2)
plt.show()
plt.clf()


------------------------------------------------------------------------------
# ppf
# binom.ppf(q, n, p)
# 개념
## 퍼센트 포인트 함수 (Percent Point Function) 
## 이것은 주어진 확률 q에 해당하는 값을 찾아줍니다.
## 쉽게 말해, 누적 확률이 q가 되는 가장 작은 성공 횟수를 반환.


binom.ppf(0.5, n=30, p=0.26)

# 예시1

## 30번 시도 중 각 시도에서 성공 확률이 26%일 때, 
## 누적 확률이 50%가 되는 최소 성공 횟수를 반환.
## "동전을 30번 던졌을 때 앞면이 나올 확률이 50%인 횟수는 몇 번일까?"라는 질문에 대한 답 

binom.ppf(0.7, n=30, p=0.26)

# 예시2
# 30번 시도 중 각 시도에서 성공 확률이 26%일 때,
# 누적 확률이 70%가 되는 최소 성공 횟수를 반환.
## "동전을 30번 던졌을 때 앞면이 나올 확률이 70%인 횟수는 몇 번일까?"라는 질문에 대한 답

binom.cdf(8, n=30, p=0.26)

# 30번 시도 중 각 시도에서 성공 확률이 26%일 때,
# 성공 횟수가 8번 이하일 확률을 계산합니다.
# 즉, "동전을 30번 던졌을 때 앞면이 8번 이하로 나올 확률은 얼마일까?"라는 질문에 대한 답

binom.cdf(9, n=30, p=0.26)

# 30번 시도 중 각 시도에서 성공 확률이 26%일 때, 
# 성공 횟수가 9번 이하일 확률을 계산합니다.
# 즉, "동전을 30번 던졌을 때 앞면이 9번 이하로 나올 확률은 얼마일까?"라는 질문에 대한 답

---------------------------------------------------------------------------------------

#정규분포 수식 
#pdf

1/np.sqrt(2 * math.pi) 
from scipy.stats import norm

# x=확률변수, loc=평균(mu), scale=표준편차 (sigma)- 중심에서 퍼진 정도 
                                                    (# sigma **2 : 분산)
norm.pdf(0, loc=0, scale=1) 
norm.pdf(5, loc=3, scale=4) 

k = np.linspace(-3, 3, 100)
y = norm.pdf(np.linspace(-3, 3, 100), loc=0, scale=1) 

plt.scatter(k, y, color="red", s=1)
plt.show()


##mu(loc) - 중심 결정 (모평균)
k = np.linspace(-5, 5, 100)
plt.plot(k, y, color="black")
plt.show()
plt.clf()

##sigma (scale) - 분포의 퍼짐 결정하는 모수 (모표준편차)
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1) 
y2 = norm.pdf(k, loc=0, scale=2)
y3 = norm.pdf(k, loc=0, scale=0.5) # 평균 근처에서 퍼짐이 작음 

plt.plot(k, y, color="black")
plt.plot(k, y2, color="red")
plt.plot(k, y3, color="blue")
plt.show()
plt.clf()


# norm.cdf

# "왼쪽에서부터" 특정값까지의 넓이
norm.cdf(0, loc=0, scale=1) #0
norm.cdf(100, loc=0, scale=1) #1

# p(-2<x<0.54) =?
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

# p(x<1 or x>3) =?
norm.cdf(1, loc=0, scale=1)
1 - norm.cdf(3, loc=0, scale=1)

# x~N(3,5^2) 3-mu 5^2-sigma
# p(3<x<5) = ? 15.54%
norm.cdf(5,3,5) - norm.cdf(3,3,5)

# 위 확률 변수에서 표본 1000개를 뽑아보자
x = norm.rvs(loc = 3, scale = 5, size = 1000 )  
sum((x > 3) & (x < 5))


# 평균 0, 표준편차 1
# 표본 1000개 뽑아서 0보다 작은'비율' 확인
x = norm.rvs(loc = 0, scale =1, size = 1000 )
type(x)
np.mean(x < 0)
(x < 0).mean()
len(x < 0)
sum(x < 0) # mean이랑 똑같이 

# 히스토그램
x = norm.rvs(loc = 3, scale = 2, size = 1000 )
x
sns.histplot(x, stat = "density")
plt.show()
plt.clf()
x = norm.rvs(loc = 3, scale = 2, size = 1000)
sns.histplot(x, stat = 'density')

xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc = 3, scale = 2)
plt.plot(x_values, pdf_values, color = 'red')
plt.show()
plt.clf()


