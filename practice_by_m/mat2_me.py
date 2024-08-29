# 회귀분석 데이터행렬
import numpy as np

x=np.array([13, 15,
           12, 14,
           10, 11,
           5, 6]).reshape(4, 2)
x
vec1=np.repeat(1, 4).reshape(4, 1)
matX=np.hstack((vec1, x)) # 합치기 
y = np.array([20, 19, 20, 12]).reshape(4, 1)
y
matX

# minimize로 라쏘 베타 구하기 
# 라쏘- 변수 선택 능력이 있다( how?-람다를 잘 사용해야함 )
from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta) #(n,1)
    return (a.transpose() @ a) + 500 * np.abs(beta[1:]).sum() 
#  beta[1:] : 절편(Intercept) 항에 대해서는 라쏘 페널티(L1 규제)를 적용하지 않음.
#  np.abs(beta[1:]).sum()  = 놈1
#  return 값이 작아질수록 좋은 회귀계수


line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([3.76,  1.36, 0]) 

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


[8.55, 5.96, -4.38] # 람다 0 > 선형회귀랑 동일 
# 예측식: y_hat = 8.55 + 5.96 * X1 + (-4.38) * X2

[8.14, 0.96, 0] # 람다 3
# 예측식: y_hat = 8.14 + 0.96 * X1 + 0 * X2

[17.74, 0, 0] # 람다 500
# 예측식: y_hat = 17.74 + 0 * X1 + 0 * X2

# 람다 값에 따라 변수 선택된다. 
# X 변수가 추가되면, trainx에서는 성능 항상 좋아짐
# x 변수가 추가되면, valid 에서는 좋아졌다가 나빠짐(오버피팅)
# 어느순간 x 변수 추가하는 건을 멈춰야함
# 람다 0 부터 시작 : 내가 가진 모든 변수를 넣겠다
# 람다를 점점 증가 : 변수가 하나씩 빠지는 효과 / 필요한 정보만 남기겠다..
# valid x 에서 가장 성능이 좋은 람다를 선택!
# 변수가 선택됨을 의미. 


# (X^T X)^-1
# X의 칼럼에 선형 종속인 애들 있다 : 다중공선성이 존재한다.























