# 더미 활용 
# 기존 숫자형 데이터와 같이 문자형도 볼 수 있게 됨 

# "Neighborhood" 열을 기준으로 원-핫 인코딩을 수행
# 원-핫 인코딩(One-Hot Encoding):
# pd.get_dummies: 범주형 변수(예: "Neighborhood")를 여러 개의 이진(0 또는 1) 열로 변환. 
# 각 열은 해당 범주의 존재 여부를 나타냅니다.
# drop_first=True: 첫 번째 범주를 삭제하여 다중공선성 문제를 방지합니다. 
# 이 옵션을 사용하면 첫 번째 카테고리 대신 나머지 카테고리들에 대한 이진 변수를 생성합니다. 
# 예를 들어, 원래 'A', 'B', 'C'라는 세 가지 범주가 있으면 'B', 'C'에 대한 이진 변수만 남게 됩니다. 
# 'A'는 기본적으로 다른 변수가 모두 0일 때 나타나는 것으로 간주됩니다.

# 범주형 변수(Categorical Variable)
# 데이터에서 범주(또는 그룹)를 나타내는 변수로, 숫자보다는 명칭이나 레이블로 표현됩니다

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
# house_train=house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임

# len(house_train["Neighborhood"].unique()) # 동네 25개 
house_train["Neighborhood"]

neighborhood_dummies = pd.get_dummies(
    house_train["Neighborhood"],  
    drop_first=True
    )
neighborhood_dummies

# 숫자형 데이터와 합치기 
# pd.concat([df_a, df_b], axis=1) ch) np.concatanate
x= pd.concat([house_train[["GrLivArea", "GarageArea"]], 
             neighborhood_dummies], axis=1)
x
y = house_train["SalePrice"]
y

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# ======

# 테스트 셋에 적용하기 
neighborhood_dummies_test = pd.get_dummies(
    house_test["Neighborhood"],
    drop_first=True
    )
neighborhood_dummies_test

test_x= pd.concat([house_test[["GrLivArea", "GarageArea"]], 
                   neighborhood_dummies_test], axis=1)
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df ["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)


