# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용 
# 종속 변수 : bill_length_mm

import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 펭귄 데이터 로드
penguins = load_penguins()
penguins.head()

# 결측치 제거
df = penguins.dropna()

# 종속변수 : bill_length_mm 를 y로 지정
df = df.rename(columns={"bill_length_mm": "y"})

# 범주형 변수 더미 코딩
df = pd.get_dummies(df, drop_first=True)

# 입력 변수와 출력 변수 분리
X = df.drop(columns='y') # 독립 변수는 2차원 형태로 제공되어야 함
y = df['y'] # 종속변수 : 예측 대상 

# 학습 데이터와 테스트 데이터로 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------
# 엘라스틱 넷 회귀 모델 생성 및 학습
## 라쏘(Lasso)와 리지(Ridge) 회귀라는 두 가지 규제(페널티)를 결합한 모델
## 오버피팅 방지, 페널티를 적용해서 더 일반화된(다양한 데이터에서 잘 작동하는) 모델 생성.

## 모델 생성
from sklearn.linear_model import ElasticNet
model = ElasticNet()

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

param_grid={
    'alpha': np.arange(0, 0.2, 0.01),
    'l1_ratio': np.arange(0.8, 1, 0.01)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(X,y)

grid_search.best_params_ #alpha=0.19, l1_ratio=0.99 #최적의 파라미터를 출력
grid_search.cv_results_ #교차 검증(Cross Validation) 동안 얻은 모든 결과
grid_search.best_score_ #가장 성능이 좋은 모델의 점수를 출력
best_model = grid_search.best_estimator_  # 가장 좋은 모델을 저장

# param_grid :조정하고 싶은 두 가지 하이퍼파라미터(모델의 설정 값)를 지정
# alpha : 람다(패널티) 
  ## 모델에 규제를 얼마나 강하게 적용할지/값이 클수록 규제가 더 강함
# l1_ratio : 알파(라쏘 가중치) 
# ## 라쏘와 리지 규제의 비율을 조절/ 0에 가까울수록 리지 규제가 많고, 1에 가까울수록 라쏘 규제가 많아요.
# GridSearchCV()**는 여러 하이퍼파라미터 값을 시도해보고 가장 좋은 모델을 찾는 도구

# scoringe-모델의 성능을 평가하는 방법을 설정하는 옵션
# scoring='neg_mean_squared_error'모델의 성능을 평가할 때 음수로 변환된 MSE를 사용하겠다는 뜻. 
# MSE는 작을수록 좋으니까, 음수로 바꾸면 큰 값이 좋은 성능을 의미


# 예측
pred_y = best_model.predict(X_test)

# 모델 성능 평가
mse = mean_squared_error(y_test, pred_y) # 예측한 값과 실제 값 사이의 평균 제곱 오차를 계산
r2 = r2_score(y_test, pred_y) # 결정계수(R-squared) /이 값이 1에 가까울수록 모델이 데이터를 잘 설명하고 있다는 의미

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# -----------------------------------------------------

# 디시전 트리 회귀 모델 생성 및 학습
# 나무 구조를 기반으로 데이터를 분류하거나 예측하는 모델
# 데이터를 여러 기준에 따라 나누고, 각 그룹에서 평균값을 구해 예측을 수행합니다.


# 모델 생성
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor(random_state=42)
param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

# 하이퍼파라미터 튜닝

grid_search2=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search2.fit(X,y)

grid_search2.best_params_ #8, 22
grid_search2.cv_results_
grid_search2.best_score_
best_model=grid_search2.best_estimator_

model2 = DecisionTreeRegressor(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model2.fit(X,y)

from sklearn import tree
tree.plot_tree(model)

# 예측
pred_y2 = best_model.predict(X_test)

# 모델 성능 평가
mse_2 = mean_squared_error(y_test, pred_y2) # 예측한 값과 실제 값 사이의 평균 제곱 오차를 계산
r2_2 = r2_score(y_test, pred_y2) # 결정계수(R-squared) /이 값이 1에 가까울수록 모델이 데이터를 잘 설명하고 있다는 의미

print(f"Mean Squared Error: {mse_2}")
print(f"R-squared: {r2_2}")



# 성능 평가는 **평균 제곱 오차(MSE)**와 **결정계수(R-squared)**로 측정
# MSE는 작을수록 좋고, R-squared는 1에 가까울수록 좋아요.