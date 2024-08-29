### house_price 데이터의 모든 범주형 칼럼들을 더미코딩해볼까?

# 하지만, 변수들 중에서 good 이나 very good처럼 순서가 있는 아이들은 
# 숫자로 바꿔줘야하고, 숫자로 되어있음에도 불구하고 
# 범주형인 데이터도 있을 것이다. 이런 친구들도 더미코딩을 해 줘야한다. 
# 이런 경우 우리들이 변수를 보고 수정을 해야하지만, 
# 시간이 없으니까 object 타입 열만 가져와서 해보자.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import os
# cwd = os.getcwd()

# 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

# NaN 채우기
# 각 숫자변수는 평균채우기
# 각 변수형변수는 최빈값 채우기
house_train.isna().sum()
house_test.isna().sum() 

# 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True) # inplace=True는 원본 데이터프레임을 직접 수정
house_train[quant_selected].isna().sum()

# 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True) # 최빈값으로 채운다며?
house_train[qual_selected].isna().sum()


house_train.shape
house_test.shape
train_n = len(house_train)
            
# 통합 df 만들기 + 더미 코딩 (- 더미 한꺼번에 진행하려고

df = pd.concat([house_train, house_test],ignore_index = True)
df.select_dtypes(include=[object]).columns
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df 

# train/test 데이터셋
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]
# train_y = y[:train_n]

# validation 셋 (모의고사 셋) 만들기
# 1460 * 0.3
# np.random.randint(0, 1459, size = 438) > 겹친 값 나올 수 있음 
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438, replace = False)
val_index

# train >>> valid / new train 데이터셋 
valid_df = train_df.loc[val_index]   # 30%
train_df = train_df.drop(val_index)  # 70%

# 이상치 탐색 
train_df=train_df.query("GrLivArea <= 4500")

# 사용할 변수 합치기 
# x = pd.concat([df[["GrLivArea", "GarageArea"]], 
#            neighborhood_dummies], axis=1)
# y = df["SalePrice"]

# x, y 나누기
# regex (Regular Expression, 정규방정식)
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
# selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df.drop["SalePrice", axis = 1]
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df.drop["SalePrice", axis = 1]
valid_y=valid_df["SalePrice"]

## test
test_x=test_df.drop["SalePrice", axis = 1]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 () - 오차니까 작을수록 좋음 
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))

## test 셋 결측치 채우기
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)


# ===================
# myf(n) 넣으면 n차 성능 나오는 함수 코드 

def myf(n):
    np.random.seed(42)
    x = uniform.rvs(size=30, loc=-4, scale=8)
    y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)
    df = pd.DataFrame({"x": x, "y": y})
    model = LinearRegression()

    train_df = df.loc[:19]
    test_df = df.loc[20:]

    name = []
    for i in range(1, n + 1):
        train_df["x" + str(i)] = train_df["x"]**i
        test_df["x" + str(i)] = test_df["x"]**i
        name.append("x" + str(i))
    
    train_x = train_df[name]
    y = train_df["y"]
    model.fit(train_x, y)

    test_x = test_df[name]
    y_hat = model.predict(test_x)
    return sum((test_df["y"] - y_hat)**2)