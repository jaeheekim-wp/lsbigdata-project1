import numpy as np

# 벡터 * 벡터 (내적)
a = np.arange(1, 4)
b = np.array([3, 6, 9])

a.dot(b)

# 행렬 * 벡터 (곱셈)
a = np.array([1, 2, 3, 4]).reshape((2, 2),
                                   order='F')
a

b = np.array([5, 6]).reshape(2, 1)
b

a.dot(b)
a @ b
## @ 로 대체 가능 

# 행렬 * 행렬
a = np.array([1, 2, 3, 4]).reshape((2, 2),
                                   order='F')
b = np.array([5, 6, 7, 8]).reshape((2, 2),
                                   order='F')
a
b
a @ b

# Q1.
a = np.array([1, 2, 1, 0, 2, 3]).reshape(2, 3)
b = np.array([1, 0, -1, 1, 2, 3]).reshape(3, 2)

a @ b

# Q2
np.eye(3)
a=np.array([3, 5, 7,
            2, 4, 9,
            3, 1, 0]).reshape(3, 3)

a @ np.eye(3)
np.eye(3) @ a

#단위행렬
## 바뀌어도 동일


# transpose 행렬 뒤집기
a
a.transpose()
b=a[:,0:2]
b
b.transpose()

# 회귀분석 데이터행렬
x=np.array([13, 15,
           12, 14,
           10, 11,
           5, 6]).reshape(4, 2) # 펭귄 네마리
x

vec1=np.repeat(1, 4).reshape(4, 1) # y 절편 값
matX=np.hstack((vec1, x))
y=np.array([20, 19, 20, 12]).reshape(4, 1)
matX

beta_vec=np.array([2, 0, 1]).reshape(3, 1)
beta_vec

matX @ beta_vec # 이거를 model.predict 가 해주는 것

(y - matX @ beta_vec).transpose() @ (y - matX @ beta_vec) # 손실함수

# 역행렬
a=np.array([1, 5, 3, 4]).reshape(2, 2)
a_inv=(-1/11)*np.array([4, -5, -3, 1]).reshape(2, 2)

a @ a_inv
#a = np.array([[1,5],[3,4]])
#b = 1/(4-15) * np.array([4,-5,-3,1]).reshape(2,2)
#a@b

# 역행렬 구하는 함수 
# np.linalg.inv(a)

## 3 by 3 역행렬
a=np.array([-4, -6, 2,
            5, -1, 3,
            -2, 4, -3]).reshape(3,3)
a_inv=np.linalg.inv(a)
np.linalg.det(a)
a_inv

np.round(a @ a_inv, 3)

## 역행렬 존재하는 않는 경우(선형종속)

# 역행렬은 항상 존재하지 않음
# 행렬의 세로 백터들이 선형독립일때만, 역행렬을 구할 수 있음.
# 선형 독립이 아닌 경우 (선형종속 )
# 이를 특이행렬이라고 부름 sigular matrix

b=np.array([1, 2, 3,
            2, 4, 5,
            3, 6, 7]).reshape(3,3)
b_inv=np.linalg.inv(b) # 에러남
np.linalg.det(b) # 행렬식이 항상 0

# ===============================
# 01벡터 형태로 베타 구하기
matX
y
XtX_inv=np.linalg.inv((matX.transpose() @ matX))
Xty=matX.transpose() @ y
beta_hat=XtX_inv @ Xty
beta_hat

# ===============================
# 02모델 fit으로 베타 구하기
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(matX[:, 1:], y) # x변수만 넣어주려고 베타제로 뺌.


model.intercept_
model.coef_

# ===============================
# 03minimize로 베타 구하기
from scipy.optimize import minimize

def line_perform(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta) # y_hat = matX @ beta
    return (a.transpose() @ a) # 제곱값이 나와 ! 그 회귀 마이너스 없애는 방식 중 하나 

line_perform([ 8.55,  5.96, -4.38])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess) # 적은 오차값이여야하니까

# 결과 출력
print("최소값:", result.fun)  # ex부리길이
print("최소값을 갖는 x 값:", result.x) # ex부리깊이/날개길이


# =======================================
# minimize로 라쏘 베타 구하기

# beta 값을 최적화하여, 예측 오차를 줄이면서도 
# 복잡한 모델을 피하는 라쏘 회귀를 수행

from scipy.optimize import minimize

def line_perform_lasso(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta).sum() #3=람다 / np.abs(beta).sum() : norm1

line_perform([8.55,  5.96, -4.38])
line_perform([3.76,  1.36, 0])
line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([3.76,  1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0] # 세 개의 회귀계수(베타값) 모두 0에서 시작

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# =======================================
# minimize로 릿지 베타 구하기
from scipy.optimize import minimize

def line_perform_ridge(beta):
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3*(beta**2).sum()

line_perform_ridge([8.55,  5.96, -4.38])
line_perform_ridge([3.76,  1.36, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# initial_guess: 베타 값의 초기 추측값 (처음엔 [0, 0, 0]으로 시작).
 ## 최적화 알고리즘: 시작점에서 출발해 손실 함수가 줄어드는 방향으로 조금씩 이동.
 ## 최솟값 발견: 더 이상 손실 함수가 줄어들지 않는 지점에서 최솟값을 발견
 
# minimize 함수: 손실 함수를 최소화하는 베타 값을 찾음.
# result.fun: 최솟값 (손실 함수의 최소값).
# result.x: 최솟값을 만드는 베타 값들 (최적의 회귀 계수).









 














