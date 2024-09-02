
# 라이브러리 호출
import pandas as pd
import numpy as np

# 확률분포표 작성
probability = pd.DataFrame({'x' : [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                            'prob' : [1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36 ,1/36]})
probability

# Q1. E[X] =?, Var(X) = ?, # 7, 5.8333
E_x = sum(probability['x'] * probability['prob'])
E_x

# Var(x) = E(x^2) - (E_x)^2
probability['x^2'] = probability['x'] ** 2 
Var = sum(probability['x^2'] *  probability['prob']) - (E_x)**2
Var

# Q2. 2X+3의 기댓값, 표준편차? # 17, 4.83
# E[2x+3]
probability['2x+3'] = 2 * probability['x'] + 3
E_2x_plus_3 = sum(probability['2x+3'] * probability['prob'])
E_2x_plus_3

# Var(2x+3)
probability['(2x+3)^2'] = probability['2x+3'] ** 2
Var_2x_plus_3 = sum(probability['(2x+3)^2'] * probability['prob']) - (E_2x_plus_3)**2
Var_2x_plus_3

# std(2x+3)
std_2x_plus_3 = np.sqrt(Var_2x_plus_3)
std_2x_plus_3

# =======================

import numpy as np
import pandas as pd

# 1
X = np.arange(2, 13, 1)
p = np.array([1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])

E = sum(X * p)
V = sum((X ** 2) * p) - (E ** 2)

# 2
X2 = (2 * X) + 3

E2 = sum(X2 * p)
V2 = sum((X2 ** 2) * p) - (E2 ** 2)
std = np.sqrt(V2)

# =======================

from scipy.stats import binom
sum(binom.pmf(np.arange(7,15), 20, 0.45))
binom.cdf(14, 20, 0.45) - binom.cdf(6, 20, 0.45)

# P(X > 24) =?
from scipy.stats import norm
1- norm.cdf(24, loc = 30, scale = 4)

# ===================
#표본 8개 뽑고 표본 평균 X_VAR
#P(28 < X_VAR < 29.7) =?

# x_var ~ N (30, 4^2/8)
a = norm.cdf(29.7, loc = 30, scale = np.sqrt(4**2/8))
b = norm.cdf(28, loc = 30, scale = np.sqrt(4**2/8))
a-b

# 결언니 코드 

# 표본을 8개 뽑아서 표본평균 X_bar 계산
# P(28 < X_bar < 29.7) = ?

sample_8 = norm.rvs(loc=30, scale=4, size=8, random_state = 2024)
X_bar_E = sum(sample_8) / len(sample_8)
X_bar_V = sum((sample_8 - X_bar_E)**2)/len(sample_8)
X_bar_s = np.sqrt(X_bar_V)

# 표본 X_bar는 N(X_bar_E, X_bar_s ^ 2)인 정규분포를 따른다.
norm.cdf(29.7, X_bar_E, X_bar_s) -  norm.cdf(28, X_bar_E, X_bar_s)



# 자유도 7인 카이제곱분포 확률밀도 함수 그리기
from scipy.stats import chi2
import matplotlib.pyplot as plt

k = np.linspace(-2, 40, 500)
y = chi2.pdf(k, df = 7)
plt.plot(k, y, color="black")


mat_a = np.array([14, 4, 0, 10]).reshape(2,2)
mat_a

# 카이제곱 검정이라는 통계 분석을 해주는 함수
from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(3)
p
expected

#출력 값들: 이 함수는 4가지 값을 출력해:

# chi2 (카이제곱 통계량): 두 변수 간의 관계가 있는지를 판단하는 데 사용하는 숫자야. 값이 클수록 두 변수가 관계가 있을 가능성이 커.
# p-value: 이 값은 그 관계가 우연히 발생할 확률을 말해줘. 이 값이 작을수록 (일반적으로 0.05보다 작을 때), 두 변수 간의 관계가 통계적으로 유의미하다고 판단해.
# df (자유도): 이건 통계 계산에서 사용하는 숫자야. 데이터의 차원을 나타내는 개념이야.
# expected: 만약 두 변수가 서로 완전히 독립적이라면 기대되는 값(예상되는 값)을 계산해줘