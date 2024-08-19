
# 숫자형 변수들 전체 활용 

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train = pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df = pd.read_csv("./data/houseprice/sample_submission.csv")

house_train.info()

# 01.트레인데이터 

# 숫자형 변수만 추가하기 
x = house_train.select_dtypes(include = [int, float])

# 필요없는 칼럼 제거 -id / saleprice
x = x.iloc[:, 1:-1] 
y = house_train["SalePrice"]

# 결측치
x.isna().sum()
x['LotFrontage'] = x['LotFrontage'].fillna(x["LotFrontage"].mean())
x['MasVnrArea'] = x['MasVnrArea'].fillna(x["MasVnrArea"].mean())
x['GarageYrBlt'] = x['GarageYrBlt'].fillna(x["GarageYrBlt"].mean())

# 강사샘 방식 
# 변수별로 결측값 채우기-데이터프레임화

# fill_values = {
#     "LotFrontage" : x["LotFrontage"].mean(), # numpy
#     "MasVnrArea" : x["MasVnrArea"].mean(), # 시리즈로 나와서 가져옴
#     "GarageYrBlt" : x["GarageYrBlt"].mean()
# }
# 
# x = x.fillna(value=fill_values)
# x.isna().sum()
# x.mean()


# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 예측값 계산
pred_y = model.predict(x)

-----------
# 02.테스트데이터

# 숫자형 데이터 불러오기 
test_x = house_test.select_dtypes(include=[int, float])

# 필요없는 변수 제거 - id 
test_x = test_x.iloc[:, 1:]
test_x.info()

# 결측치 제거 
test_x.isna().sum()
test_x = test_x.fillna(test_x.mean()) 

# for i in test_x.columns:
#     test_x[i].fillna(test_x[i].mean(),inplace=True)  #용규 코드  
    
 ## inplace=True - 원래 데이터프레임인 test_x를 직접 수정/ 새로운 시리즈 생성 x

-----------
# 테스트 데이터 집값 예측 
pred_y = model.predict(test_x)

sub_df["SalePrice"] = pred_y 
sub_df.to_csv("./data/houseprice/sample_submission0805.csv", index=False)


