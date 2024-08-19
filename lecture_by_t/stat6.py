# y=2x+3 그래프 그리기
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# x 값의 범위 설정
x = np.linspace(0, 100, 400)

# y 값 계산
y = 2 * x + 3

# np.random.seed(20240805)
obs_x=np.random.choice(np.arange(100), 20)
epsilon_i=norm.rvs(loc=0, scale=100, size=20)
obs_y = 2*obs_x+3 + epsilon_i

# 그래프 그리기
plt.plot(x, y, label='y = 2x + 3', color="black")
plt.scatter(obs_x, obs_y, color="blue", s=3)
# plt.show()
# plt.clf()

from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
obs_x=obs_x.reshape(-1,1)
model.fit(obs_x, obs_y)

# 회귀 직선의 기울기와 절편
model.coef_[0]      # 기울기 a hat
model.intercept_ # 절편 b hat

# 회귀 직선 그리기
x = np.linspace(0, 100, 400)
y = model.coef_[0] * x + model.intercept_
plt.xlim([0, 100])
plt.ylim([0, 300])
plt.plot(x, y, color="red") # 회귀직선
plt.show()
plt.clf()

model
summary(model)

# -----------------------

# !pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y,obs_x).fit()
print(model.summary())
 # 기울기 / 절편 값 포함한 다양한 값 제공
 
# 검정 통계량 ---------------------------------------------

# mu = 10 (귀무가설)
# mu != 10 (대립가설)
 ## n = 20 / x_bar = 18
# sigma = 8.79
# sd =  8.79 / np.sqrt(20) # 1.96
# 18 이후가 나올 확률
norm.cdf(18, loc =10, scale = 1.96)
1- norm.cdf(18, loc =10, scale = 1.96) 
# p-value 유의확률 -
# 엄청 작은 확률인데도 불구하고 이 값이 나왔어?  - 귀무가설 기각, 못믿어!

------------------------------------------------------
# hw6

# 귀무가설 H0 mu >=16 > 1등급 부여 
# 대립가설 Ha mu < 16

x = np.array([15.078, 15.752, 15.549, 15.56, 16.098, 13.2771, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,
15.382, 16.709, 16.804])

n =15
mu0 = 16

# 표본평균
x_mean = x.mean() #15.531

# 표본표준편차 s
x_std = np.std(x, ddof=1)

# t계산하기- 검정통계량
## t = (x_bar - mu0) / (s/np.sqrt(n))

## 이 값은 두 그룹 간의 '평균 차이'를 표준오차로 나눈 결과. 
## t 검정통계량은 두 그룹의 차이가 우연히 발생할 확률이 낮은지를 판단하는 데 사용. 
## 일반적으로 t 검정통계량의 절대값이 클수록 두 그룹 간의 차이가 통계적으로 유의미할 가능성이 큽니다.

t_value = (np.mean(x)-16)/(np.std(x, ddof=1)/np.sqrt(15))
t_value

# 유의확률 p-value 구하기 
t.cdf(t_value, df = n-1) #  0.04%

# 유의수준 1% 와 비교해서 기각 결정 
0.01 < 0.04 
# 이 값이 발견될 확률은 유의 수준보다도 높으니 
# 기각하지 못한다 >>> 1등급 부여 


# 현대자동차의 신형 모델의 평균 복합 에너지 소비효율에 대하여 95% 신뢰구간
# 신뢰구간 계산: 평균 ± (z-값 * SE)
# SE = std/ sqrt(n)
ci_1 = x_mean + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(15)
ci_2 = x_mean - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(15)


# X_bar 부분의 sig_2 / n에서 n은 모표본의 n
# t 은 표본(파란벽돌)이라서 n-1 
