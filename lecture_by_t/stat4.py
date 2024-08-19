import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import seaborn as sns
# !pip install scipy

# uniform 28p

# X ~ 균일분포 U(a, b)
# loc: a, scale: b-a
# 균일 분포
# loc은 구간시작점, scale은 구간 길이 ! 주의 
# loc = a / scale = b-a

# uniform.pdf(x, loc=0, scale=1)
# uniform.cdf(x, loc=0, scale=1)
# uniform.ppf(q, loc=0, scale=1)


# uniform.rvs(loc=0, scale=1, size=None, random_state=None)

uniform.rvs(loc=2, scale=4, size=1)
uniform.pdf(3, loc=2, scale=4)
uniform.pdf(7, loc=2, scale=4)

# 중요: 균등 분포에서는 [2, 6] 구간 외부의 값에 대해 확률 밀도는 0. 
# 따라서, uniform.pdf(7, loc=2, scale=4)의 결과는 0. 
# 이는 7이 [2, 6] 구간에 포함되지 않기 때문입니다.

k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color="black")
plt.show()
plt.clf()

# p(x<3.25 )= ?
uniform.cdf(3.25, loc=2, scale=4) 
# 균일 분포에서 특정 값 x까지의 누적 확률(q)을 계산

# p(5<x<8.39) = ?
uniform.cdf(8.39, loc=2, scale=4) -  uniform.cdf(5, loc=2, scale=4) 
uniform.cdf(6, loc=2, scale=4) -  uniform.cdf(5, loc=2, scale=4) 
# 6까지의 균일 분포니까 동일값

# 상위 7%
uniform.ppf(0.93, loc=2, scale=4)

-----------------------------------------------------------------------

# 예제 : 표본 20개 뽑고 표본평균 계산

# X~U(2,6) 
# 01.모분산에서 표본 20개 뽑기 

x = uniform.rvs(loc=2, scale=4, size=20*1000,
              random_state=42) 
x = x.reshape(-1, 20)

# random_state = random.seed
# rvs 사용해 20000개의 표본 뽑고 
# reshape(1000,20) 사용하여 20개 표본 뽑는 것을 1000번 반복하겠다는 의미. 

x.shape
# (1000, 20)

# 02.표본 평균 구하기
# (x1 + x2  +x3 +...+ x20 = x_bar = blue_x ) >>> 총 1000개의 행 
blue_x = x.mean(axis=1) # 20개씩 더해 평균 내기 
blue_x
blue_x.shape


# 03.표본 평균 그래프 

sns.histplot(blue_x, stat="density")
plt.show()
plt.clf()



# 모분산( X~U(2,6) )
uniform.var(loc=2, scale=4)  # 1.33333    # uniform의 loc은 구간 시작!
# 기댓값
uniform.expect(loc=2, scale=4) # 4.0

-------------------------------------------------------------------

# 표본 평균의 정규 분포 PDF 그리기 

# X bar ~ N(mu, sigma^2/n)  
# X bar ~ N(4, 1.3333333/20)  
  ## 4표본분산은 [모분산/ n ]  즉, [1.3333333/20] 가 표본 분산
                     
# E[x] = mu 니까 loc = 4

xmin, xmax = (blue_x.min(), blue_x.max()) # 그래프 X축 범위 지정.
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, 
                      scale= np.sqrt(1.3333333/20)) # 표본표준편차는 표본분산의 제곱근 
                                                   
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()


# 신뢰구간 그리기 

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.3333333/20) 평균이 4이고 표본 분산이 [1.3333333/20]
from scipy.stats import norm

# PDF 겹쳐 그리기 
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4,
                      scale= np.sqrt(1.3333333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()


# 표본 평균(파란벽돌) 점찍기
blue_x = uniform.rvs(loc=2, scale=4, size=20).mean()  
blue_x 
# 왜loc이 4가 아니고 2?- # X~U(2,6) 에서 가져오는 표본들이니까!


# 95% 신뢰구간 표시하기 (자료 38p)
# norm.ppf

## 신뢰구간 계산
## 1. 표준오차(SE) 계산: 모표준편차를 샘플 크기의 제곱근으로 나눕니다
    # SE = 모표준편차 / √샘플 크기
## 2. z-값 사용: 95% 신뢰구간에서는 z-값이 1.96입니다 
    # (정확히는 norm.ppf(0.975, loc=0, scale=1)에서 나온 값).
## 3. 신뢰구간 계산: 평균 ± (z-값 * SE)

norm.ppf(0.975, loc=0, scale=1) #  1.96( Z값 )
norm.ppf(0.025, loc=0, scale=1) # -1.96( Z값 )

a = blue_x + 1.96 * np.sqrt(1.3333333/20) # np.sqrt(1.3333333/20) 이 표준편차 아닌가, 표준 오차는 편차를 나누라며
b = blue_x - 1.96 * np.sqrt(1.3333333/20)
plt.scatter(blue_x, 0.002, 
            color='blue', zorder=10, s=10)
            
# 신뢰 구간 파란 라인 
plt.axvline(x=a, color='blue', 
            linestyle='--', linewidth=1)
plt.axvline(x=b, color='blue', 
            linestyle='--', linewidth=1)
norm.ppf(0.995, loc=0, scale=1)

# 기대값 표현
plt.axvline(x=4, color='green', 
            linestyle='-', linewidth=2)

plt.show()
plt.clf()


# norm.ppf 함수 설명
# 신뢰구간 계산에 어떻게 사용되는지 

# norm.ppf(0.975, loc=0, scale=1)
# 표준 정규분포에서 상위 97.5% 지점의 값을 찾는 것. 
# 표준 정규분포는 평균이 0이고 표준편차가 1인 분포입니다.

# 0.975는 누적분포함수(CDF)에서의 확률 값을 의미합니다. 97.5%의 지점.
# loc=0은 평균이 0/ scale=1은 표준편차가 1임을 의미.
# 이 값을 찾는 이유는 95% 신뢰구간을 계산할 때, 양쪽 끝에서 2.5%씩 잘라내기 때문입니다. 
# 따라서, 상위 97.5% 지점의 값을 사용하는 것이죠.
    

4 - norm.ppf(0.025, loc=4, scale= np.sqrt(1.3333333/20))
4 - norm.ppf(0.975, loc=4, scale= np.sqrt(1.3333333/20))

import seaborn as sns

sns.histplot(blue_x, stat="density")
plt.show()


