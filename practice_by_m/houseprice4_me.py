 # 조별 회기분석 예측 
 # 변수  "두 개" 사용해서 회귀모델을 만들고 제출할 것.
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#필요한 데이터 불러오기
house_test = pd.read_csv('data/houseprice/test.csv')
house_train = pd.read_csv('data/houseprice/train.csv')
sub_df = pd.read_csv('data/houseprice/sample_submission.csv')

#이상치 탐색 및 제거 
house_train.query("GrLivArea > 4500") #탐색
house_train = house_train.query("GrLivArea <= 4500") #4500보다 작거나 같은 것만 할당해줌
 ## house_train['GrLivArea'].sort_values(ascending = False).head(2)

# 회귀분석 적합(FIT)하기 
# x = np.array(house_train[["GrLivArea", "GarageArea"]]).reshape(-1, 2) 
x = house_train[["GrLivArea", "GarageArea"]] 
y = np.array(house_train["SalePrice"]) 

# LinearRegression 사용할때 입력데이터가 2차원배열이여야 함.
# house_train["GrLivArea"] -> 판다스 시리즈형태 = 넘파이벡터(1차원), 차원 없음
# house_train[["GrLivArea"]] -> 판다스 데이터프레임형태 = 2차원 벡터형태

   
# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # fit함수가 자동으로 기울기, 절편 값을 구해줌.

# 회귀 직선의 기울기와 절편
model.coef_         # 기울기 a
model.intercept_    # 절편 b

slope = model.coef_
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산 
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()

# -----------------------------------------

f(x,y) = ax + by + c
def my_houseprice(x, y):
    return model.coef_[0] * x +  model.coef_[1]*y + model.intercept_
my_houseprice(300, 55)

# model.predict(test_x) 한거랑 동일 


house_test["GrLivArea"]
house_test["GarageArea"]
my_houseprice(house_test["GrLivArea"], house_test["GarageArea"])

# 결측치 
house_test = house_test.fillna(house_test["GarageArea"].mean())
test_x = np.array(house_test[["GrLivArea","GrLivArea"]]).reshape(-1,2)

test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()

pred_y = model.predict(test_x) 
len(pred_y)

#SalePrice 바꿔치기
sub_df["saleprice"] = pred_y
sub_df

#csv로 바꿔치기
sub_df.to_csv("data/houseprice/sample_submission9.csv", index = False)

--------------------



