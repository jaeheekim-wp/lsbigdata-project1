# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!
# 원하는 변수 2개
# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

## 이상치 탐색
house_train=house_train.query("GrLivArea <= 4500")
house_train

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임

# 타겟 변수 선택 
x = house_train[["GrLivArea", "GarageArea"]] # 거실면적/ 차고 면적 
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a  # 기울기 값 두개인 이유..? > 변수 두개!
model.intercept_ # 절편 b

slope = model.coef_
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# f(x,y) = ax + by + c
# def my_houseprice(x, y):
#     return model.coef_[0]*x + model.coef_[1]*y + model.intercept_
# 
# my_houseprice(300, 55) 
# my_houseprice(house_test["GrLivArea"], house_test["GarageArea"]) 

# model.predict(test_x) 과 동일 


# 테스트 파일 불러오기 
test_x = house_test[["GrLivArea", "GarageArea"]]
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x = test_x.fillna(house_test["GarageArea"].mean())

# 예측값 계산
pred_y = model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기

sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission7.csv", index=False)

------------------------------------


# 3차원 평면 그래프 만들기 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# 데이터 로드
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

# 이상치 제거
house_train = house_train.query('GrLivArea <= 4500')

# 피처와 타겟 변수 선택
x = house_train[['GrLivArea', 'GarageArea']]
y = house_train['SalePrice']

# 모델 학습
model = LinearRegression()
model.fit(x, y)

# 기울기와 절편 출력
slope_grlivarea = model.coef_[0]
slope_garagearea = model.coef_[1]
intercept = model.intercept_

print(f"GrLivArea의 기울기 (slope): {slope_grlivarea}")
print(f"GarageArea의 기울기 (slope): {slope_garagearea}")
print(f"절편 (intercept): {intercept}")

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트
ax.scatter(x['GrLivArea'], x['GarageArea'], y, color='blue', label='Data points')

# 회귀 평면
GrLivArea_vals = np.linspace(x['GrLivArea'].min(), x['GrLivArea'].max(), 100)
GarageArea_vals = np.linspace(x['GarageArea'].min(), x['GarageArea'].max(), 100)
GrLivArea_vals, GarageArea_vals = np.meshgrid(GrLivArea_vals, GarageArea_vals)
SalePrice_vals = intercept + slope_grlivarea * GrLivArea_vals + slope_garagearea * GarageArea_vals

ax.plot_surface(GrLivArea_vals, GarageArea_vals, SalePrice_vals, color='red', alpha=0.5)

# 축 라벨
ax.set_xlabel('GrLivArea')
ax.set_ylabel('GarageArea')
ax.set_zlabel('SalePrice')

plt.legend()
plt.show()
plt.clf()



