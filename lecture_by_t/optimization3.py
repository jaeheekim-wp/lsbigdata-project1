# 회귀직선 베타 찾기

import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점(9,2)에 파란 점 표시 
plt.scatter(9, 2, color = 'red', s=50)

x = 9; y = 2 
lstep = 0.1
x, y = np.array([x, y]) - lstep * np.array([2*x-6, 2*y-8])

for i in range(100) : 
    (x, y) = np.array([x, y]) - lstep * np.array([2*x -6, 2*y-8])
    plt.scatter(x,y,color="red", s=50)
print(x,y)

# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# ========================================

# x, y의 값을 정의합니다 (-1에서 7까지)
beta0 = np.linspace(-20, 20, 400)
beta1 = np.linspace(-20, 20, 400)
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(x, y)를 계산합니다.
z = (1-(beta0+beta1))**2 + (4-(beta0+2*beta1))**2 + (1.5-(beta0+3*beta1))**2 + (5-(beta0+4*beta1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=100)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점(9,2)에 파란 점 표시 
plt.scatter(9, 9, color = 'red', s=100)

# f(beta0, beta1) = (1-(beta0+beta1))**2 + (4-(beta0+2*beta1))**2 + (1.5-(beta0+3*beta1))**2 + (5-(beta0+4*beta1))**@
beta0 = 10
beta1 = 10
delta = 0.01
for i in range(1000):
    gradient_beta0 = 8*beta0 + 20*beta1 -23
    gradient_beta1 = 20*beta0 + 60*beta1 -67
    beta0, beta1 = np.array([beta0, beta1]) - delta * np.array([gradient_beta0, gradient_beta1])
    plt.scatter(beta0, beta1, color = 'red', s=10)
print(beta0, beta1)


# ==========================
# 회귀직선 베타 찾기
import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
beta0 = np.linspace(-20, 20, 400)
beta1 = np.linspace(-20, 20, 400)
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(x, y)를 계산합니다.
z = (1-(beta0+beta1))**2 + (4-(beta0+2*beta1))**2 + (1.5-(beta0+3*beta1))**2 + (5-(beta0+4*beta1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=100)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (9, 9)에 파란색 점을 표시
plt.scatter(9, 9, color='red', s=100)

beta0=9; beta1=9
lstep=0.01
for i in range(1000):
    beta0, beta1 = np.array([beta0, beta1]) - lstep * np.array([8*beta0+20*beta1-23, 20*beta0 + 60*beta1-67])
    plt.scatter(float(beta0), float(beta1), color='red', s=25)

print(beta0, beta1)

# 축 레이블 및 타이틀 설정
plt.xlabel('beta0')
plt.ylabel('beta1')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# 그래프 표시
plt.show()


# 모델 fit으로 베타 구하기
import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.DataFrame({
    'x': np.array([1, 2, 3, 4]),
    'y': np.array([1, 4, 1.5, 5])
})
model = LinearRegression()
model.fit(df[['x']], df['y'])

model.intercept_
model.coef_
