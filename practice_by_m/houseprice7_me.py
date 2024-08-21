# 더미 활용 
# 기존 숫자형 데이터와 같이 문자형도 볼 수 있게 됨 

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
#house_train=house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임

# len(house_train["Neighborhood"].unique())
neighborhood_dummies = pd.get_dummies(
    house_train["Neighborhood"],
    drop_first=True
    )

neighborhood_dummies
# pd.concat([df_a, df_b], axis=1)
x= pd.concat([house_train[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies], axis=1)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

neighborhood_dummies_test = pd.get_dummies(
    house_test["Neighborhood"],
    drop_first=True
    )
    
# 테스트에 적용    
test_x = pd.concat([house_test[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies_test], axis=1)
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x = test_x.fillna(house_test["GarageArea"].mean())

pred_y = model.predict(test_x) # test 셋에 대한 집값
len(pred_y)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)

# 시각화
# 직선값 계산
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
