import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# houseprce
# 범주형 변수로 회귀분석 진행하기

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

# 동네별 GRLIV  VS SALEPRICE 
fig = px.scatter(
    house_train,
    x="GrLivArea",
    y="SalePrice",
    color="Neighborhood",
    trendline="ols" # p.134 선형회귀직선 그려줌 (개별 기울기,절편 )
)

fig.show()

# 레이아웃 업데이트 -dict()로 중괄호 대체 가능

fig.update_layout(
    title=dict(text="GrLivArea vs. SalePrice", font=dict(color="#667BC6")),
    paper_bgcolor="#FDFFD2",  # 전체 배경색
    plot_bgcolor="#F1D3CE", # 플롯 영역 배경색
    font=dict(color="#667BC6"),
    xaxis=dict(
        title=dict(text="GrLivArea", font=dict(color="#667BC6")),  # x축 제목 글씨 색상
        tickfont=dict(color="#667BC6"), # x축 눈금 글씨 색상:
        gridcolor='white'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="SalePrice", font=dict(color="#667BC6")), 
        tickfont=dict(color="#667BC6"),
        gridcolor='white'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="#667BC6")),
)

# 점 크기 및 투명도 설정
fig.update_traces(marker=dict(size=10, opacity = 0.7))

# 시각화 
fig.show()

# ================
# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 Neighborhood  를 더미 변수로 변환

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
neighborhood_dummies.

# 숫자형 데이터와 합치기 
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
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
# sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)

# --------------------------------
# 회기직선 시각화 
import matplotlib.pyplot as plt
import seaborn as sns

# 종 별 데이터  
sns.scatterplot(x=test_x["GrLivArea"], y=y, 
                hue=house_test["Neighborhood"], palette="deep",
                legend=False)
# 회기직선 ( 기울기,절편 통일 )                
sns.scatterplot(x=test_x["GrLivArea"], y=pred_y,size = 2,
                color="black")
plt.show()
plt.clf()

# ===================================================


