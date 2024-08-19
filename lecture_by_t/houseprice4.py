# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!

# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## 이상치 탐색
house_train =house_train.query("GrLivArea <= 4500")
 ## house_train['GrLivArea'].sort_values(ascending = False).head(2)
 
# ====
## 회귀분석 적합(fit)하기 
x = np.array(house_train["GrLivArea"]).reshape(-1, 1)
y = house_train["SalePrice"]

# LinearRegression 사용할때 입력데이터가 2차원배열이여야 함.
# house_train["GrLivArea"] -> 판다스 시리즈형태 = 넘파이벡터(1차원), 차원 없음
# house_train[["GrLivArea"]] -> 판다스 데이터프레임형태 = 2차원 벡터형태

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# ====

# 테스트 불러오기 
test_x = np.array(house_test["GrLivArea"]).reshape(-1, 1)
test_x

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y
pred_y=pred_y/1000

# =====

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission6.csv", index=False)

# ====

# 시각화
# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, pred_y, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 2000])
plt.ylim([0, 300])
plt.legend()
plt.show()
plt.clf()

# -----------------------------
f(x,y) = ax + by + c
def my_houseprice(x, y):
    return model.coef_[0] * x +  model.coef_[1]*y + model.intercept_
my_houseprice(300, 55)

# model.predict(test_x) 한거랑 동일 

