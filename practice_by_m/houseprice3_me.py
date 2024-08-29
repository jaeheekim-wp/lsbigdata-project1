import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 직선의 방정식 
# y = ax +b 
# ex) y = 2X+3
a = 1
b = 3

x = np.linspace(-5,5,100)
y = a * x + b
plt.plot(x, y, color="blue")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.show()
plt.clf()

# --------------
# y = x
a = 1
b = 3

x = np.linspace(-5,5,100)
y = x
plt.plot(x, y, color="red")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.show()
plt.clf()


# b : 절편은 들어올리는 기능 'y 절편'
# a : (곱해진 수) 는 기울기 조절

# --------------

a = 50
b = 50

x = np.linspace(0, 5, 100)
y = a * x + b

house_train = pd.read_csv("./data/houseprice/train.csv")
my_df = house_train[["BedroomAbvGr","SalePrice"]].head(10)
my_df["SalePrice"] = my_df["SalePrice"] /1000
plt.scatter(data = my_df, x = 'BedroomAbvGr', y = 'SalePrice')
plt.plot(x, y, color = "blue")
plt.show()
plt.clf()

---------------

house_test = pd.read_csv("./data/houseprice/test.csv")
a = 63,  b = 100
(a * house_test["BedroomAbvGr"] + b) *1000

# sub 데이터 불러오기 
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# 바꿔치기 
sub_df["saleprice"] = (a * house_test["BedroomAbvGr"] + b) *1000
sub_df

# 직선 구하기
sub_df.to_csv("./data/houseprice/sample_submission2.csv", index=False)
sub_df

# 직선 성능 평가

a = 70
b = 10

# y_hat 어떻게 구할까?
y_hat = (a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디에 있는가?
y = house_train["SalePrice"]


np.abs(y - y_hat) # 절대거리 
np.sum(np.abs(y - y_hat)) # 절대값 합 
np.sum(np.abs((y - y_hat)**2)) # 제곱

-----------------------------------------------------------------------

# !pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression() # 제곱방식으로 모델을 생성해줌 

# 모델 학습 - 계산해주는 
model.fit(x, y) # 자동으로 기울기, 절편 구해줌 

# 회귀 직선의 기울기와 절편 - 이미 계산된 값을 꺼내주는 
slope = model.coef_[0]       # 기울기 a
intercept = model.intercept_ # 절편 b
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

------------------------------------

# houseprice 활용

house_train = pd.read_csv("data/houseprice/train.csv")
my_df = house_train[["BedroomAbvGr", "SalePrice"]]


# 선형 회귀 모델 생성
x = np.array(my_df["BedroomAbvGr"]).reshape(-1,1) # 시리즈 불가.
                                                  # 넘파이 어레이에서만 가능( reshape도np에서만 ) 
y = my_df["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 데이터와 회귀 직선 시각화
model.coef_
model.intercept_

# 회귀 직선의 기울기와 절편
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

my_df['SalePrice'] =  y_pred
sub = pd.read_csv("data/houseprice/sample_submission.csv")
sub["SalePrice"] = my_df['SalePrice']

sub.to_csv("sub_prediction085.csv", index=False)

------------------------------------------

