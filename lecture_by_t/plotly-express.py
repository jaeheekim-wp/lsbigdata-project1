# 데이터 패키지 설치
# !pip install palmerpenguins
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# x: bill_length_mm
# y: bill_depth_mm  
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    trendline="ols"  # p.134 선형회귀직선 그려줌 (개별 기울기,절편 )
)

fig.show()

# 레이아웃 업데이트 -dict()로 중괄호 대체 가능

# 1.제목 크기 키울것,
# 2.점 크기 크게, 
# 3.범례 제목 "펭귄 종" 으로 변경

fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white")),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(text="펭귄 종",font=dict(color="white")),
)

# 점 크기 및 투명도 설정
fig.update_traces(marker=dict(size=10, opacity = 0.7))

fig.show()

# ==========================

# 132p 선형회귀모델
from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins=penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

model.fit(x, y)
linear_fit = model.predict(x)

model.coef_
model.intercept_

# 선형 회귀 추세선 추가 - 이미 그려진 그래프에 추가해라 
fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot", color="white")
    )
)
fig.show()

# 심슨의 역설(Simpson's paradox) 
# 데이터의 세부 그룹별로 일정한 추세나 경향성이 나타나지만, 
# 전체적으로 보면 그 추세가 사라지거나 반대 방향의 경향성을 나타내는 현상. 
# 사회과학이나 의학 통계 연구에서 종종 발생한다. 
# 심슨의 역설은 통계의 함정이 유발할 수 있는 잘못된 결과를 설명하는 데 쓰이기도 한다.

# ======================================

# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환

penguins = load_penguins()

penguins_dummies = pd.get_dummies(
    penguins, 
    columns=['species'],
    drop_first=True
    )
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_ # array([ 0.20044313, -1.93307791, -5.10331533])
model.intercept_  # np.float64(10.565261622823762)

regline_y=model.predict(x)

# 시각화 

import matplotlib.pyplot as plt
import seaborn as sns

# 종 별 데이터  
sns.scatterplot(x=penguins["bill_length_mm"], y=y, 
                hue=penguins["species"], palette="deep",
                legend=False)
# 회기직선 ( 기울기,절편 통일 )                
sns.scatterplot(x=penguins["bill_length_mm"], y=regline_y,size = 2,
                color="black")
plt.show()
plt.clf()

# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
# penguins
# species    island  bill_length_mm  ...  body_mass_g     sex  year
# Adelie     Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap  Torgersen            40.5  ...       3800.0  female  2007
# Gentoo     Torgersen            40.5  ...       3800.0  female  2007
# x1, x2, x3
# 39.5, 0, 0 > Adelie는 F/F로 넣으면 분석 가능 
# 40.5, 1, 0 >
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56 노트 필기 참고 
0.2 * 40.5 -1.93 * True -5.1* False + 10.56


# > house price 7 으로 이동 
