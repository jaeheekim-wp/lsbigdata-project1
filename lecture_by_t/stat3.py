
# !pip install scipy
from scipy.stats import bernoulli
from scipy.stats import binom
from scipy.stats import norm

import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt

import seaborn as sns


# 01.pmf (확률 질량 함수): 특정 횟수만큼 성공할 확률을 계산
# 02.pdf (확률 밀도 함수): 연속형 확률 분포에서 특정 값이 나올 확률의 '밀도' 계산


# 01.pmf (확률 질량 함수): 특정 횟수만큼 성공할 확률을 계산
# 02.pdf (확률 밀도 함수): 연속형 확률 분포에서 특정 값이 나올 확률의 밀도 계산
# 03.rvs (랜덤 샘플 함수): 이항 분포를 따르는 난수를 생성
# 04.cdf (누적 분포 함수): 특정 횟수 이하로 성공할 확률을 계산
# 05.ppt (역 누적 분포 함수): 누적 확률이 q가 되는 가장 작은 성공 횟수를 반환

# binom
# 이항 분포란?
# 이항 분포는 일련의 독립적인 시행에서 각 시행이 성공 또는 실패, 
# 두 가지 결과만을 가지는 경우, 성공의 횟수를 나타내는 <확률 분포>.
# 즉, 고정된 횟수의 독립적인 시행(예: 동전 던지기)에서 성공 횟수를 모델링 (앞면 나오는)


# binom 함수를 사용하여 이항 분포와 관련된 계산
  ## 이산형 분포(ex.0,1,2...)
  ## 매개변수:n(시행 횟수) p(성공 확률)
  
  ## ch) bernoulli 베르누이 분포 : 단일 분포
  
  ## ch) norm 정규 분포 : 
         ### 연속성 분포 : 결과가 연속적인
         ### 매개변수: mean(loc 평균/데이터중심값),std( scale 표준편차)
  
  #사용 함수:
  #binom.pmf(k, n, p): k 번 성공할 확률.
  #binom.cdf(k, n, p): k 번 이하로 성공할 확률.
  #binom.rvs(n, p, size): 이항 분포를 따르는 랜덤 값을 생성.


# pmf
# 확률 질량 함수
# 특정 횟수만큼 성공할 확률을 계산
# probability mass function
a = np.array(1,2,3)
a = np.array([])
# 각 가능한 값에 확률을 할당하는 함수
# bernoulli.pmf(k, p)
# P(X=1)

bernoulli.pmf(1, 0.3)
# P(X=0)
bernoulli.pmf(0, 0.3)

# bernoulli
# 베르누이 시행이란?
# 베르누이 시행은 결과가 딱 두 가지뿐인 실험이나 시도.

## 동전을 던져서 앞면이 나오는지 뒷면이 나오는지
## 시험에 합격하는지 불합격하는지
## 특정 제품이 불량인지 아닌지

# 베르누이 분포란?
# 베르누이 분포는 단일 시도에서 두 가지 결과(성공 또는 실패)만 있을 때 사용

# ex) 동전을 한 번 던져서 앞면이 나오거나 뒷면이 나오는 경우.
  ## bernoulli.pmf(k, p)는 성공 확률 p인 베르누이 실험에서 
  ## 성공(k=1) 또는 실패(k=0)의 확률을 구합니다.
  
# bernoulli.pmf 함수
# bernoulli.pmf 함수는 베르누이 분포에서 특정 결과가 나올 확률을 계산  

# rvs
# 표본 '추출' 함수
# 이항 분포를 따르는 난수를 생성
# random variates sample

# X1 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
norm.rvs(loc=3,scale=2,size=3)
# X2 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)

# X ~ B(n=2, p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)

binom.rvs(n=2, p=0.3, size=1)
# binom.rvs으로 랜덤값(표본-0,1,2) 추출 

binom.pmf(0, n=2, p=0.3)  #0.49        
binom.pmf(1, n=2, p=0.3)  #0.42
binom.pmf(2, n=2, p=0.3)  #0.09

# -------------------------------------------------------------------

binom.pmf(x, n, p)

# binom
# 이항 분포란?
# 이항 분포는 일련의 독립적인 시행에서 각 시행이 성공 또는 실패, 
# 두 가지 결과만을 가지는 경우, 성공의 횟수를 나타내는 <확률 분포>.

## binom.pmf(x, n, p)
## 성공 확률 p(전체 앞면)인 베르누이 실험을 n번 반복했을 때, 
## 성공이 k번 일어날'확률'을 구합니다.


# 이항 분포와 관련된 계산
  ## 이산형 분포(ex.0,1,2...)
  ## cf) bernoulli 베르누이 분포 : 단일 분포

  ## cf) norm 정규 분포 : 
         ### 연속성 분포(결과가 연속적인)
         ### 매개변수: mean(loc 평균/ 데이터중심값),std( scale 표준편차)

# binom.pmf로 확률값 추출 


binom.pmf(x, n, p)

## 성공 확률 p(전체 앞면)인 베르누이 실험을 n번 반복했을 때, 
## 성공이 k번 일어날'확률'을 구합니다.
 
 # 이항 분포의 주요 파라미터:
 # x: 성공한 횟수(앞면이 나오는 횟수)
 # n: 시행의 총 횟수
 # p: 각 시행에서 성공할 확률

binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# 설명 :
# n = 2는 동전을 2번 던지는 것과 같고,
# p = 0.3는 동전이 앞면(1)이 나올 확률이 30%인 것.
# 이제 우리가 계산하는 것은 동전을 2번 던졌을 때 앞면이 몇 번 나오는지.

# (binom.pmf(0, 2, 0.3)):
# 앞면이 한 번도 안 나올 확률
# 동전을 2번 던졌을 때 2번 모두 뒷면이 나올 확률입니다.
# 확률은 약 49%입니다.

# (binom.pmf(1, 2, 0.3)):
# 앞면이 한 번 나올 확률
# 동전을 2번 던졌을 때 1번은 앞면, 1번은 뒷면이 나올 확률입니다.
# 확률은 약 42%입니다.

# (binom.pmf(2, 2, 0.3)):
# 앞면이 두 번 다 나올 확률 
# 동전을 2번 던졌을 때 2번 모두 앞면이 나올 확률입니다.
# 확률은 약 9%입니다.


# 정리 
## pmf (Probability Mass Function): 특정 값이 나올 확률을 나타내는 함수

## 베르누이 분포의 PMF (bernoulli.pmf): 
 ### 단일 시도에서 성공 또는 실패의 확률을 계산.
 
## 이항 분포의 PMF (binom.pmf):
 ### 여러 번의 시도에서 특정 횟수만큼 성공할 확률을 계산.
 
 
# binom.rvs: 동전을 던졌을 때 무작위로 나올 결과를 시뮬레이션한다.
# binom.pmf: 동전을 던졌을 때 특정 결과가 나올 확률을 계산한다.

# 사용 함수:
# binom.pmf(k, n, p): k 번 성공할 확률.
# binom.cdf(k, n, p): k 번 이하로 성공할 누적 확률.
# binom.rvs(n, p, size): 이항 분포를 따르는 랜덤 값을 생성.


## PMF (Probability Mass Function): 특정 값이 나올 확률을 나타내는 함수

## 베르누이 분포의 PMF (bernoulli.pmf): 
 ### 단일 시도에서 성공 또는 실패의 확률을 계산.
## 이항 분포의 PMF (binom.pmf):
 ### 여러 번의 시도에서 특정 횟수만큼 성공할 확률을 계산.
 
 ## PMF는 각 가능한 값에 확률을 할당해줌으로써, 
 ##특정 값이 나올 가능성을 알 수 있다.
 

# binom.rvs: 동전을 던졌을 때 무작위로 나올 결과를 시뮬레이션한다.
# binom.pmf: 동전을 던졌을 때 특정 결과가 나올 확률을 계산한다.
# ----------------------------------
# 예제

# X ~ B(30, 0.26)
# 기대값 30*0.26 = 7.8

# 표본 30개를 뽑아보세요!
binom.rvs(n=30, p=0.26, size=1)   # 파라미터는 모집단
binom.rvs(n=30, p=0.26, size=30)   # 사이즈는 랜덤 샘플 값값
#- -----------------------------------
# 기대값 30*0.26 = 7.8-

# 표본 30개를 뽑아보세요!
binom.rvs(n=30, p=0.26, size=10)   # 파라미터는 모집단
binom.rvs(n=30, p=0.26, size=30)   
# ----------------------------------

# X ~ B(n, p)
a = np.arange(0,31,2)
a = np.linspace(0,30,15)

# list comp.
result=[binom.pmf(x, n=30, p=0.3) for x in range(31)]
result

for x in range(31):
  result.append(binom.pmf(x,n=30,p=0.3))

# [(앞면이 0번 나올 확률), (앞면이 1번 나올 확률), ..., (앞면이 30번 나올 확률)]

# numpy
binom.pmf(np.arange(31), n=30, p=0.3)

# math 
math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54, 26)

# ======몰라도 됨==================================

import numpy as np
binom.pmf(np.arange(31), n=30, p=0.3)

# math 
import math
math.factorial(54) / (math.factorial(26) * math.factorial(28))
math.comb(54, 26)

# ======몰라도 됨====================================

# 1*2*3*4
# np.cumprod(np.arange(1, 5))[-1]
# fact_54=np.cumprod(np.arange(1, 55))[-1]
# ln
log(a * b) = log(a) + log(b)
log(1 * 2 * 3 * 4) = log(1) + log(2) + log(3) + log(4)

np.log(24)
sum(np.log(np.arange(1, 5)))

math.log(math.factorial(54))
logf_54=sum(np.log(np.arange(1, 55)))
logf_26=sum(np.log(np.arange(1, 27)))
logf_28=sum(np.log(np.arange(1, 29)))
# math.comb(54, 26)
np.exp(logf_54 - (logf_26 + logf_28))
# =================================================

math.comb(2, 0) * 0.3**0 * (1-0.3)**2
math.comb(2, 1) * 0.3**1 * (1-0.3)**1
math.comb(2, 2) * 0.3**2 * (1-0.3)**0


# 예제1
# X ~ B(n=10, p=0.36)

# P(X = 4) =? 
binom.pmf(4, n=10, p=0.36)  

# P(X <= 4) =?
binom.pmf([0,1,2,3,4], 10, 0.36).sum()
binom.pmf(np.arange(5), n=10, p=0.36).sum()

# 확률변수 x가 2보다 크고, 8보다 작거나 같을 확률을 구하시오.
# P(2 < X <= 8)
binom.pmf(np.arange(3, 9), n=10, p=0.36).sum()



# 예제2
# X ~ B( n = 30, p = 0.2) 
binom.pmf(np.arange(31), n=30, p=0.2).sum()
# 성공 확률이 0.36인 사건을 10번 시도했을 때, 
# 성공 횟수가 3번 이상 8번 이하일 확률을 구하고 그 확률을 모두 더한 값을 반환
# 이 값을 계산하면, 해당 범위 내에서의 성공 확률을 알 수 있습니다.

# X ~ B(n = 30, p = 0.2) 
binom.pmf(np.arange(31), n=30, p=0.2).sum()

# 확률변수 x가 4보다 작거나, 25보다 크거나 같을 확률을 구하시오.
# P(X<4 or X>=25)?
# 1
a = binom.pmf(np.arange(4), n=30, p=0.2).sum()
b = binom.pmf(np.arange(25, 31), n=30, p=0.2).sum()
a+b

# 2 
# 확률의 총 합은 100, 즉 1이라는 개념 활용
# 1- P(4<=X<25)?
1 - binom.pmf(np.arange(4, 25), n=30, p=0.2).sum()

# 시각화 

import seaborn as sns
import matplotlib.pyplot as plt
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
# prob_x
# [P(앞면이 0번 나올 확률), P(앞면이 1번 나올 확률), ..., P(앞면이 30번 나올 확률)]

# binom.pmf(0, n=30, p=0.26) >>pro_x에 나오는 값 맞는지 확인용 

sns.barplot(prob_x)
plt.show()
plt.clf()

# 교재 p.207

import pandas as pd

x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)

# 데이터 프레임화

df = pd.DataFrame({"x": x, "prob_x": prob_x})
df

sns.barplot(data = df, x = "x", y = "prob_x")

df = pd.DataFrame({"x": x, "prob": prob_x})
df

sns.barplot(data = df, x = "x", y = "prob")
plt.show()

-----------------------------------------------------------------------

# cdf
# 누적 확률 분포 함수 
# 특정 횟수 이하로 성공할 확률
# 특정 횟수 이하로 성공할 확률을 계산
# cumulative dist. function
# F_X(x) = P(X <= x)

binom.cdf(4, n=30, p=0.26)

# 상황 :
# 당신은 동전을 30개 던집니다.
# 이 동전은 전체 앞면이 나올 확률이 26%입니다.
# 우리는 동전을 30개 던졌을 때 앞면이 4번 이하로 나올 확률이 궁금합니다.

# 풀이 :
# 이 함수는 동전을 30번 던졌을 때 
# 앞면이 0번, 1번, 2번, 3번, 4번 나올 각각의 확률을 모두 더합니다.
# 즉, 앞면이 4번 이하로 나올 확률을 계산합니다.

# 예시
# binom.cdf(4, n=30, p=0.26)는 다음 확률들을 더한 값.
# 동전을 30개 던졌을 때 앞면이 0번 나올 확률+
# 동전을 30개 던졌을 때 앞면이 1번 나올 확률+
# 동전을 30개 던졌을 때 앞면이 2번 나올 확률+
# 동전을 30개 던졌을 때 앞면이 3번 나올 확률+
# 동전을 30개 던졌을 때 앞면이 4번 나올 확률

# 4 < x <=18
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26) 

# 동전을 30개 던졌을 때 앞면이 0번 나올 확률
# 동전을 30개 던졌을 때 앞면이 1번 나올 확률
# 동전을 30개 던졌을 때 앞면이 2번 나올 확률
# 동전을 30개 던졌을 때 앞면이 3번 나올 확률
# 동전을 30개 던졌을 때 앞면이 4번 나올 확률

4<x <=18
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26) ?

# 시각화

import numpy as np
import seaborn as sns

x_1 = binom.rvs(n=30, p=0.26, size=10)
x_1 = binom.rvs(n=30, p=0.26, size=1)

x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color="blue")

# Add a point at (2, 0)
plt.scatter(x_1, 0.2, color='red', zorder=100, s=10)

# 기대값 표현
# E[X] = n * p 
# E[X] = 30 * 0.26 

plt.axvline(x=7.8, color='green', 
            linestyle='--', linewidth=2)
# 세로선 그리기 ch) axhline 가로 라인 
plt.show()
plt.clf()

# 기대값이 7.8이 나오는 이유-일준님
aa = np.arange(31)
bb = binom.pmf(np.arange(31), n = 30, p = 0.26)
sum(aa * bb)

# ppf
# 퍼센트 포인트 함수/ 역 누적 분포 함수

plt.scatter(x_1, 0.002, color='red', zorder=100, s=10)

# 기대값 표현
# E[X] = n * p 
# E[X] = 30 *0.26  ## 7.8이 되는 원리는 이해 못했음..
plt.axvline(x=7.8, color='green', 
            linestyle='--', linewidth=2)

plt.show()
plt.clf()

# ppf
# 역 누적 분포 함수/ 퍼센트 포인트 함수

# 누적 확률이 q가 되는 가장 작은 성공 횟수를 반환
# Percent Point Function
# binom.ppf(q, n, p)

# 예시
a = binom.ppf(0.5, n=30, p=0.26)
a

# 예시1
binom.ppf(0.5, n=30, p=0.26)

## 30번 시도 중 각 시도에서 성공 확률이 26%일 때, 
## 누적 확률이 50%가 되는 최소 성공 횟수를 반환.
## "동전을 30개 던졌을 때 앞면이 나올 확률이 50%인 횟수는 몇 번일까?"라는 질문에 대한 답 

# 예시2
binom.ppf(0.7, n=30, p=0.26)

# 30번 시도 중 각 시도에서 성공 확률이 26%일 때,
# 누적 확률이 70%가 되는 최소 성공 횟수를 반환.
## "동전을 30번 던졌을 때 앞면이 나올 확률이 70%인 횟수는 몇 번일까?"라는 질문에 대한 답

# cdf 활용 

# 예시3
binom.cdf(8, n=30, p=0.26)

# 30번 시도 중 각 시도에서 성공 확률이 26%일 때,
# 성공 횟수가 8번 이하일 확률을 계산합니다.
# 즉, "동전을 30번 던졌을 때 앞면이 8번 이하로 나올 확률은 얼마일까?"라는 질문에 대한 답

# 예시4
binom.cdf(9, n=30, p=0.26)
# 30번 시도 중 각 시도에서 성공 확률이 26%일 때, 
# 성공 횟수가 9번 이하일 확률을 계산합니다.
# 즉, "동전을 30번 던졌을 때 앞면이 9번 이하로 나올 확률은 얼마일까?"라는 질문에 대한 답



# pdf
# 확률 밀도 함수

# 연속형 확률 분포에서 특정 값(x)이 나올 확률의 밀도 계산
# 연속형 확률 분포에서 특정 값이 나올 확률의 밀도 계산

# 특정 값 주변의 데이터 밀도를 계산하여, 그 값이 얼마나 흔한지/드문지

from scipy.stats import norm

# 정규분포
# Normal distribution / norm 

# 정규분포 수식 
1/np.sqrt(2 * math.pi)


# x = 확률변수 중의 한 값 
  
# loc = 모평균(== mu/mean) :
   ## 모집단의 평균값으로, 모집단의 모든 값의 합을 모집단의 크기로 나눈 값 /"중심"
  
# scale = 모표준편차 (== sigma/std) : 
   ## 모분산의 제곱근으로, 모집단의 데이터가 평균(중심)에서 벗어난 정도/"분포의 폭"
   ### 모분산 (==sigma^2): 모집단의 데이터가 평균에서 얼마나 떨어져 있는지를 나타내는 척도.

# scale = 모표준편차 (== sigma) : 
   ## 모분산의 제곱근으로, 모집단의 데이터가 평균(중심)에서 벗어난 정도/"분포의 폭"
   ### 모분산 (==sigma^2) 모집단의 데이터가 평균에서 얼마나 떨어져 있는지를 나타내는 척도.


norm.pdf(0, loc=0, scale=1)
norm.pdf(5, loc=3, scale=4)
 # 평균 = 3 표준 편차 = 4인 정규 분포
 # 즉, 중심이 3이고, 4만큼의 폭을 가진 종 모양의 분포
 
 # 정규 분포는 무한이다 


# 시각화하기 
# pdf 그리기

k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)

 # 평균이 0이고 표준 편차가 1인 정규 분포에서 k의 각 값에 대한 '확률 밀도'를 계산.
 # 즉, 이 정규 분포 곡선에서 배열 k의 각 위치에서의 높이


## mu (loc): 분포의 중심 결정하는 모수 (평균)
k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=2)

## mu (loc): 분포의 중심 결정하는 모수

k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)
plt.plot(k, y, color="black")
plt.show()
plt.clf()

## sigma (scale): 분포의 퍼짐 결정하는 모수(표준편차)
k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)
y2=norm.pdf(k, loc=0, scale=2)
y3=norm.pdf(k, loc=0, scale=0.5) # 평균 근처에서 퍼짐이 작음 

plt.plot(k, y, color="black")
plt.plot(k, y2, color="red")
plt.plot(k, y3, color="blue")
plt.show()
plt.clf()


# 산의 높이 비유:
 # k 배열은 산을 따라 -5에서 5까지의 위치를 나타냅니다.
 # norm.pdf(k, loc=0, scale=1)은 이 위치에서 산의 높이를 측정하는 것.
 # 평균이 0인 산의 높이는 중앙(0)에서 가장 높고, 양쪽 끝(-5와 5)으로 갈수록 낮아집니다.
 # 이 산의 높이를 그래프로 그리면 종 모양의 곡선이 됩니다.

# norm.cdf
# 정규 분포에서 특정 값(x) 이하(<=)의 확률을 알려주는 함수
# '왼쪽'에서부터의 pdf 아래 넓이(확률)를 계산
# Norm.cdf(x, loc=, scale=) 

norm.cdf(0, loc=0, scale=1) 
norm.cdf(1, loc=0, scale=1) 
norm.cdf(10, loc=0, scale=1) 
# "왼쪽에서부터" 특정값까지의 넓이

norm.cdf(0, loc=0, scale=1) 
norm.cdf(1, loc=0, scale=1) 
norm.cdf(10, loc=0, scale=500) 

norm.cdf(100, loc=0, scale=1) 

# p(-2<x<0.54) =?
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

# p(x<1 or x>3) =?
norm.cdf(1, loc=0, scale=1)
1 - norm.cdf(3, loc=0, scale=1) 

# X ~ N(3, 5^2) 3-mu 5^2-sigma
# P(3 < X < 5) =? 15.54%
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)

# 정리 
# pdf "산의 높이를 측정하는 것"에 비유 
 ## 특정 위치에서 산이 얼마나 높은지를 알고 싶다면, 그 위치의 높이를 측정합니다. 

# cdf: "산의 높이를 쌓아 올리는 것"에 비유
 ## 특정 위치까지 쌓아 올린 모든 높이를 알고 싶다면, 그 위치까지의 모든 높이를 더합니다.

# 정리 

# pdf "산의 높이를 측정하는 것"에 비유 
 ## 특정 위치에서 산이 얼마나 높은지를 알고 싶다면, 그 위치의 높이를 측정합니다. 

# cdF: "산의 높이를 쌓아 올리는 것"에 비유
 ## 특정 위치까지 쌓아 올린 모든 높이를 알고 싶다면, 그 위치까지의 모든 높이를 더합니다.


# 정규분포: Normal distribution


# X ~ N(3, 5^2) (평균이 3, 분산이 25, 표준편차 5 )
# X ~ N(3, 5^2)
# P(3 < X < 5) =? 15.54%

# 위 확률변수에서 표본 1000개 뽑아보자!
x=norm.rvs(loc=3, scale=5, size=1000)
x
sum((x > 3) & (x < 5))/1000
np.mean((x > 3) & (x < 5))

# 평균:0, 표준편차: 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
x=norm.rvs(loc=0, scale=1, size=1000)
np.mean(x < 0)
sum(x < 0)/1000 # 동일 값

#시각화하기

x=norm.rvs(loc=3, scale=2, size=1000)
x

sns.histplot(x, stat="density")
sns.histplot(samples, bins=10, kde=False, stat='density')

# bins: 막대수, kde: 커널
# stat="density"는 히스토그램의 y축이 밀도(density)로 설정됨을 의미.


# Plot the normal distribution PDF
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)

pdf_values = norm.pdf(x_values, loc=3, scale=2) # x_values 에 해당하는 밀도 구해줘 
pdf_values = norm.pdf(x_values, loc=3, scale=2)

plt.plot(x_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()


# ---------------------
 # 평균이 30, 표준편차가 5인 분포를 따르는 확률변수 x 에서
 # 추출한 <표본크기 30인 표본들의 표본 평균>의 확률 밀도함수를 그려보세요.
 # x_bar ~ N ( mu, sigma^2/n)
 # x_bar ~ N ( mu, sigma/sqrt(n)
 # sigma^2 = 모분산 == 25
 # sigma = 모표준편차 ==5
x = np.arange(25,35,0.1)
y = norm.pdf(x, loc = 30, scale = 5/math.sqrt(30))  

# 표본표준편차 scale = 5/math.sqrt(30) = 
plt.plot(x, y, color = 'k')
plt.xlabel(" ")
plt.ylabel("p.d.f")
plt.show()

# bins: 막대수, kde: 커널 density, stat='density': 히스토그램 높이 조정
# sns.histplot(samples, bins=10, kde=False, stat='density')


# 수식 정리 

norm.pdf(x, loc=0, scale=1)
 ## 정규 분포의 특정 값 x에서의 확률 밀도를 계산
 ## 특정 값이 주어진 분포에서 얼마나 흔한지 또는 드문지
 
norm.cdf(x, loc=0, scale=1)
 ## 정규 분포에서 특정 값 x까지의 누적 확률(q)을 계산
 ## 특정 값 이하의 값이 나올 확률을 계산하는 데 사용
 
norm.ppf(q, loc=0, scale=1)
 ## 누적 확률 q에 해당하는 값(x)을 계산
 ## 정규 분포에서 특정 값 x까지의 누적 확률을 계산
 ## 특정 값 이하의 값이 나올 확률을 계산하는 데 사용
 
norm.ppf(q, loc=0, scale=1)
 ## 누적 확률 q에 해당하는 값을 계산
 ## 특정 확률에 해당하는 값을 찾는 데 사용
 
norm.rvs(loc=0, scale=1, size=None, random_state=None)
 ## 정규 분포에서 임의의 값을 생성
 ## size는 생성할 값의 개수, random_state는 시드 값
 ## 원하는 형태의 무작위 데이터를 쉽게 생성


# X_bar = 표본 평균의 분포를 따르는 확률 변수 (많이 뽑을수록 정규분포에 가까워짐)
# x_bar = 표본 평균

