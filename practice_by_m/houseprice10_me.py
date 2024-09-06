# houseprice10
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

# Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)
val_index

# train => valid / train 데이터셋
valid_df=train_df.loc[val_index]  # 30%
train_df=train_df.drop(val_index) # 70%

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기
# regex (Regular Expression, 정규방정식)
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
# selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df.drop("SalePrice", axis=1)
valid_y=valid_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# 선형 회귀 모델 생성
model = LinearRegression()

# 라쏘 모델 만들려면?
from sklearn.linear_model import Lasso
model= Lasso(alpha=0.03)

# 릿지 모델 만들려면?
from sklearn.linear_model import Ridge
model= Ridge(alpha=0.03)

from sklearn.linear_model import ElasticNet
model= ElasticNet()

param_grid={
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}

# alpha : 람다(패널티)
# l1_ratio : 알파(라쏘 가중치)

from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)

grid_search.best_params_
grid_search.cv_results_
grid_search.best_score_
best_model = grid_search.best_estimator_

best_model.predict(valid_x)


# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)


#===================
# cv_results_
# 모든 하이퍼파라미터 조합에 대해 얻은 결과를 포함하는 딕셔너리입니다. 
# 중요한 정보들이 많이 들어 있어요:

# mean_test_score: 
# 각 하이퍼파라미터 조합에 대한 평균 성능 점수 (여기서는 음수 평균 제곱 오차).
# td_test_score: 
# 각 하이퍼파라미터 조합에 대한 성능 점수의 표준 편차. 값이 작을수록 모델이 일관되게 작동하고 있다는 뜻이에요.
# params: 
# 각 하이퍼파라미터 조합. 예를 들어, alpha와 l1_ratio의 값들이 여기에 저장돼요.

# **cv_results_**의 일부를 출력하면 다음과 같은 정보를 볼 수 있어요:
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
print(results[['param_alpha', 'param_l1_ratio', 'mean_test_score', 'std_test_score']])
# 각 하이퍼파라미터 조합(alpha와 l1_ratio)와 
# 그에 대한 교차 검증 평균 점수(mean_test_score), 
# 표준 편차(std_test_score)를 출력해줘요. 
# 이 데이터를 통해 각 하이퍼파라미터 설정이 모델 성능에 어떤 영향을 미쳤는지 분석할 수 있어요.

