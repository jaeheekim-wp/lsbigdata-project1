<<<<<<< HEAD
=======
# 자격에 대하여
# 어떤 일에 대한 본인의 자격은 자신의 생각이 그렇게 
# 중요하지 않음.

# 진짜 중요한 것은 타인이 생각하는 그 자리에 대한
# 본인의 자격이 중요함.

# numpy, pandas, matplotlib 설치해볼것!
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a

import numpy as np
import matplotlib.pyplot as plt
# !pip install matplotlib

# 예제 넘파이 배열 생성
data = np.random.rand(10)
<<<<<<< HEAD
data
sum(data < 0.18)

# 히스토그램 그리기

plt.hist(data, bins=8, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.clf()

# 데이터의 분포를 시각적으로 표현하는 도구.
# 연속형 데이터의 분포 제공
=======
# sum(data < 0.18)

# 히스토그램 그리기

# 데이터의 분포를 시각적으로 표현하는 도구.
# 연속형 데이터의 분포 제.공
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
 ## 데이터의 범위를 여러 구간으로 나누고 
 ## 각 구간에 속하는 데이터의 개수를 막대 그래프로 표현한다. 
 
 ##히스토그램의 기본 개념
 ##구간(bin): 데이터를 나누는 구간. 
 ##각 구간은 동일한 너비를 가지며, 구간의 수는 사용자가 지정할 수 있습니다.
 ##빈도(frequency): 각 구간에 속하는 데이터의 개수.
 ##각 막대의 높이는 해당 구간의 빈도를 나타냅니다.
 
 ## plt.hist() 함수는 히스토그램을 그리는 함수입니다.
 ## data는 히스토그램을 그릴 데이터입니다.
 ## bins=30은 데이터를 30개의 구간으로 나누는 것을 의미합니다.
 ## alpha=0.7은 막대의 투명도를 설정합니다 (0: 완전 투명, 1: 불투명).
 ## color='blue'는 막대의 색상을 파란색으로 설정합니다.
 ## edgecolor='black'는 막대의 테두리 색상을 검정색으로 설정합니다.

<<<<<<< HEAD
=======
plt.clf()

plt.hist(data, bins=4, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a

# x=np.random.rand(50000) \
#     .reshape(-1, 5) \
#     .mean(axis=1)

x=np.random.rand(10000, 5).mean(axis=1)
<<<<<<< HEAD
x
=======
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
plt.hist(x, bins=30, alpha=0.7, color='blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

<<<<<<< HEAD
=======

>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
------------------------------------------------------------------
통계 개념 

import numpy as np

x=np.arange(33)
<<<<<<< HEAD

# E(X) 기댓값 구하기 
# sum( X가 갖는 값 * 확률)

sum(x)/33
# 즉, E(X)= 16
sum((x - 16) * 1/33)  


# 분산 Var(X)

# 첫번째 수식 
# Var(X) = E[(x - E(X))**2]

np.unique((x - 16)**2)
np.unique((x - 16)**2) * (2/33)
sum(np.unique((x - 16)**2) * (2/33))

# 두번째 수식 
# Var(X) = E[X^2] - (E[X])^2

# E[X^2]
sum(x**2 * (1/33))
=======
sum(x)/33
sum((x - 16) * 1/33)
(x - 16)**2
np.unique((x - 16)**2)

np.unique((x - 16)**2) * (2/33)
sum(np.unique((x - 16)**2) * (2/33))

# E[X^2]
sum(x**2 * (1/33))

>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
# Var(X) = E[X^2] - (E[X])^2
sum(x**2 * (1/33)) - 16**2


## Example
x=np.arange(4)
x
<<<<<<< HEAD
pro_x=np.array([1/6, 2/6, 2/6, 1/6])  # 확률
pro_x

# 기대값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x) 

# 분산 두번째 공식 계산 법 
Exx - Ex**2
# 분산 첫번째 공식 계산 법 
=======
pro_x=np.array([1/6, 2/6, 2/6, 1/6])
pro_x

# 기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x) 

# 분산
Exx - Ex**2
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
sum((x - Ex)**2 * pro_x)

## Example
x=np.arange(99)
x

# 1-50-1 벡터
<<<<<<< HEAD
x_1_50_1 = np.concatenate((np.arange(1, 51), np.arange(49, 0, -1)))
pro_x = x_1_50_1/2500

# 기대값
Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)
=======
x_1_50_1=np.concatenate((np.arange(1, 51), np.arange(49, 0, -1)))
pro_x=x_1_50_1/2500

# 기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)

sum(np.arange(50))+sum(np.arange(51))



