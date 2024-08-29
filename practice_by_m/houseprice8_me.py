# overfitting 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

# 이상치 탐색 ( 여기 위치 안댐 ) > 여기서 하면 valide에 영향을 미치게 됨 
# house_train=house_train.query("GrLivArea <= 4500")

house_train.shape
house_test.shape
train_n = house_train.shape[0]

# 통합 df 만들기 + 더미 코딩 (- 더미 한꺼번에 진행하려고
df = pd.concat([house_train, house_test],ignore_index = True)
df = pd.get_dummies(
    df,
    columns = ["Neighborhood"],
    drop_first = True   
)
df

# train/test 데이터셋 >>> 4번
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]
# train_y = y[:train_n]

# validation 셋 (모의고사 셋) 만들기
# 1460 * 0.3
# np.random.randint(0, 1459, size = 438) > 겹친 값 나올 수 있음 
# #replace = False중복 없이 선택하겠다
np.random.seed(42)
val_index = np.random.choice(np.arange(train_n), size = 438, replace = False)
val_index

# train >>> valid / new train 데이터셋 
valid_df = train_df.loc[val_index]   # 30%
train_df = train_df.drop(val_index)  # 70%
# valid_y = train_y[val_index]       # 30%
# train_y = train_y.drop(val_index)  # 70%

# 이상치 탐색 
train_df=train_df.query("GrLivArea <= 4500")

# 사용할 변수 합치기 
# x = pd.concat([df[["GrLivArea", "GarageArea"]], 
#            neighborhood_dummies], axis=1)
# y = df["SalePrice"]

# x, y 나누기
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
# 정규표현식(Regular Expression, RegEx)
# 특정한 규칙을 가진 문자열을 검색, 추출, 또는 치환하기 위해 
# 사용하는 문자열 패턴
selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df[selected_columns]
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df[selected_columns]
valid_y=valid_df["SalePrice"]

## test
test_x=test_df[selected_columns]

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
