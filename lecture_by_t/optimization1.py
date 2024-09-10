# 최적화(Optimization)란?
# 말 그대로 "최고의 선택"을 찾는 것

# 머신러닝에서 최적화란?
# 머신러닝에서 최적화는 모델의 성능을 최대한 높이는 방법을 찾는 과정
# 데이터를 학습시킬 때, 모델이 최고의 예측을 할 수 있도록 
# 모델의 파라미터(숫자)를 잘 조정하는 것이죠.

# ex)머신러닝 모델은 데이터에서 패턴을 학습해요. 이때 **"오차"**라는 것이 있어요.
# 즉, 모델이 예측한 값과 실제 값이 다를 때 그 차이를 오차라고 해요. 
# 최적화는 이 오차를 최대한 줄이는 것을 목표로 해요


# y=(x-2)^2 + 1 그래프 그리기
# 점을 직선으로 이어서 표현
import matplotlib.pyplot as plt
import numpy as np

k=2.2
x = np.linspace(-4, 8, 100)
y = (x - 2)**2 + 1
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.xlim(-4, 8)
plt.ylim(0, 15)

# f'(x)=2x-4
# k=4의 기울기
l_slope=2*k - 4
f_k=(k-2)**2 + 1
l_intercept=f_k - l_slope * k

# y=slope*x+intercept 그래프
line_y=l_slope*x + l_intercept
plt.plot(x, line_y, color="red")

# y=x^2 경사하강법
# 초기값:10, 델타: 0.9
x=10
lstep=np.arange(100, 0, -1)*0.01
for i in range(100):
    x-=lstep[i]*(2*x)

print(x)

