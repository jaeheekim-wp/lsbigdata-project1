
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# y = ax^2 + bx + c 그래프 
a = -2
b = 3
c = 5
x = np.linspace(-8, 8, 100)
y = a*x**2 + b*x +c
plt.plot(x, y, color="black")
plt.show()
plt.clf()

# y = ax^3 + bx^2 + cx + d 그래프
a = -2
b = 3
c = 5
d = -1

x = np.linspace(-8, 8, 100)
y = (a * (x ** 3)) + (b * (x ** 2)) + (c * x) + d

plt.plot(x, y, color = 'black')

# y = ax^4 + bx^3 + cx^2 + dx + e 그래프
a = 1
b = 1
c = 1
d = 1
e = 5

x = np.linspace(-8, 8, 100)
y = (a * (x ** 4)) + (b * (x ** 4)) + (c * x**2) + d*x + e
plt.plot(x, y, color = 'black')

from scipy.stats import norm
from scipy.stats import uniform

norm.rvs(size =2, loc = 0, scale = 3)

# 검정 곡선 (정답)
k = np.linspace(-4, 4, 200) # 이상적인 데이터 
sin_y = np.sin(k) 

# 파란 점들 (단서)
x = uniform.rvs(size =20, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size = 20, loc = 0, scale = 0.3) 
                ### 노이즈 개념

plt.plot(k, sin_y, color = "black")
plt.scatter(x, y, color = "blue")

# overfitting
# 과적합 방지:데이터 나누기 
# training set : 훈련 데이터 - 모델 학습 
# validation set : 검증 데이터 - 모델 평가 및 튜닝 
# test set : 테스트 데이터 - 최종 평가 

np.random.seed(42)
x = uniform.rvs(size = 30, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size = 30, loc = 0, scale = 0.3)
   ### x의 사인 값(np.sin(x))에 정규분포에서 나온 잡음을 더해 생성된 값

import pandas as pd
df = pd.DataFrame({
    "x" : x, "y" : y
})
df

train_df = df.loc[:19]
train_df

test_df = df.loc[20:]
test_df

plt.scatter(train_df["x"], train_df["y"], color="blue")

# 회귀직선 그리기
from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()
x = train_df[["x"]]
y= train_df["y"]

# 모델 학습(2차원이여야 학습 가능)
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 예측값 계산
reg_line = model.predict(x)

# 그래프 
plt.plot(x, reg_line, color='red', label='regression')
plt.scatter(train_df["x"], train_df["y"], color="blue")

# ==============================
# 2차 곡선 회귀 

train_df["x2"] = train_df["x"]**2
train_df

x = train_df[["x","x2"]] 
y = train_df[["y"]] 

model = LinearRegression()

model.fit(x, y) 

model.coef_     
model.intercept_ 

k = np.linspace(-4, 4, 200) 
df_k = pd.DataFrame({
    "x" : k, "x2" : k**2
})
df_k

reg_line = model.predict(df_k)

plt.plot(k, reg_line, color='red')
plt.scatter(train_df["x"], train_df["y"], color="blue")

# =====================================
# 3차 곡선 회귀
 
train_df["x3"] = train_df["x"]**3
train_df

x = train_df[["x","x2","x3"]]
y = train_df[["y"]] 

model.fit(x, y)

model.coef_     
model.intercept_ 

k = np.linspace(-4, 4, 200) # k:이상적인 데이터를 그리겠다 
df_k = pd.DataFrame({
    "x" : k, "x2" : k**2 , "x3" : k**3
})
df_k
reg_line = model.predict(df_k)

plt.plot(k, reg_line, color='red')
plt.scatter(train_df["x"], train_df["y"], color="blue")

# ======================================
# 4차 곡선 회귀 

train_df["x4"] = train_df["x"]**4
train_df

x = train_df[["x","x2","x3","x4"]] 
y = train_df[["y"]] 

# 모델학습 
model.fit(x, y)

# 기울기/절편
model.coef_     
model.intercept_ 

# 회귀선 예측 
# 모델에 넣어 예측할 데이터 
k = np.linspace(-4, 4, 200) 
df_k = pd.DataFrame({
    "x" : k, "x2" : k**2 , "x3" : k**3, "x4" : k**4
})
df_k
reg_line = model.predict(df_k)

plt.plot(k, reg_line, color='red')
plt.scatter(train_df["x"], train_df["y"], color="blue")

# ======================================
# 9차 곡선 회귀 

train_df["x5"] = train_df["x"]**5
train_df["x6"] = train_df["x"]**6
train_df["x7"] = train_df["x"]**7
train_df["x8"] = train_df["x"]**8
train_df["x9"] = train_df["x"]**9
train_df

x = train_df[["x","x2","x3","x4","x5","x6","x7","x8","x9"]] 
y = train_df["y"] 

model.fit(x, y) 

model.coef_     
model.intercept_ 

k = np.linspace(-4, 4, 200) 
df_k = pd.DataFrame({
    "x" : k, "x2" : k**2 , "x3" : k**3, "x4" : k**4,
    "x5" : k**5, "x6" : k**6 , "x7" : k**7, "x8" : k**8 ,"x9" : k**9
})
df_k
reg_line = model.predict(df_k)

plt.plot(k, reg_line, color='red')
plt.scatter(train_df["x"], train_df["y"], color="blue")

# test x 에 대해 예측값 구하기 
test_df['x2'] = test_df["x"] ** 2
test_df['x3'] = test_df["x"] ** 3
test_df['x4'] = test_df["x"] ** 4
test_df['x5'] = test_df["x"] ** 5
test_df['x6'] = test_df["x"] ** 6
test_df['x7'] = test_df["x"] ** 7
test_df['x8'] = test_df["x"] ** 8
test_df['x9'] = test_df["x"] ** 9

test_x = test_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]]

y_hat = model.predict(test_x)

sum((test_df['y'] - y_hat) ** 2)

# --------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자
np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "x" : x , "y" : y
})
df

# train 학습
train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
y = train_df["y"]

model=LinearRegression()
model.fit(x,y)

test_df = df.loc[20:]
test_df

for i in range(2, 21):
    test_df[f"x{i}"] = test_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = test_df[["x"] + [f"x{i}" for i in range(2, 21)]]

y_hat = model.predict(x)

# 모델 성능
sum((test_df["y"] - y_hat)**2)
