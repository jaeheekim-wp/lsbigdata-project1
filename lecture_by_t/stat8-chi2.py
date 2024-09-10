import numpy as np

mat_a = np.array([14, 4, 0, 10]).reshape(2,2)
mat_a

# 카이제곱 검정이라는 통계 분석을 해주는 함수
# 귀무가설: 두 변수 독립
# 대립가설: 두 변수가 독립 X
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

# <카이제곱 분포>
# - 독립성 검정 > chi2
# - 동질성 검정 > chi2_contingency
# - 적합도 검정 > chisquare

# [카이제곱 검정]
# - 독립성 검정 : 두 변수가 독립인지 vs. 아닌지
# - 동질성 검정 : 두 그룹별 분포가 동일한지 vs. 아닌지
# - 적합도 검정 : 데이터가 특정 분포를 따르는지 vs. 아닌지
# - 3 sample 이상 비율 검정 : p1 = p2 = p3 vs. 다른게 하나라도 있는지 > 찬성/반대

# [비율 검정 (z 검정)]
# - 1 sample : p(모비율) = p0(귀무가설이 생각하는 p값) vs. p != p0
# - 2 sample : p1 = p2 vs. p1 != p2
# - 3 sample 이상 비율 검정 : p1 = p2 = p3 vs. 다른게 하나라도 있는지 > 찬성/반대

import numpy as np
import matplotlib.pyplot as plt

# 독립성 검정
mat_a = np.array([14, 4, 0, 10]).reshape(2,2)

# 귀무가설: 두 변수(흡연, 운동선수) 독립
# 대립가설: 두 변수 독립 X
from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a, 
                                         correction=False)
# np.sum((mat_a - expected)**2/expected)
chi2.round(3) # 검정 통계량
p.round(4) # p-value

# 유의수준 0.05라면, p 값이 0.05보다 작으므로, 귀무가설 기각
# 즉, 두 변수는 독립이 아니다
# X~chi2(1) 일 때, P(X > 15.556)
from scipy.stats import chi2

1-chi2.cdf(15.556, df=1)
p

#=========================================
# 동질성 검증
mat_b=np.array([[50, 30, 20], [45, 35, 20]])

chi2, p, df, expected = chi2_contingency(mat_b, 
                                         correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value
expected

#==========================================
# p.112 연습문제
# 귀무가설: 정당 지지와 핸드폰 사용 유무는 독립이다.
# 대립가설: 정당 지지와 핸드폰 사용 유무는 독립이 아니다.
mat_phone = np.array([49, 47, 15, 27, 32, 30]).reshape(3,2)

chi2, p, df, expected = chi2_contingency(mat_phone, 
                                         correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value
# 유의수준 0.05보다 p-value가 크므로, 귀무가설을 기각할 수 없다.
expected

#==========================================
# 적합도 검정
from scipy.stats import chisquare

# 자유도는 n-1로 바뀐다(7일이니 n-1 하면 6됨)
observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp=expected)
# p-value 0.2688이 유의수준 0.05 보다 크므로 귀무가설을 기각 못함.
# 즉, 요일 별 신생아 출생비율이 같다고 판단.

from scipy.stats import chi2

1-chi2.cdf(7.6000000000000005, df=6)

#=========================================
# 지역별 후보 지지율
# 귀무가설 : 선거구별 후보A의 지지율이 동일하다.
# 대립가설 : 선거구별 후보A의 지지율이 동일하지 않다.
mat_p = np.array([[176, 124], 
                  [193, 107], 
                  [159, 141]])

from scipy.stats import chi2

chi2, p, df, expected = chi2_contingency(mat_p, 
                                         correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value
# 유의수준 0.05라면, p 값이 0.05보다 작으므로, 귀무가설 기각
expected