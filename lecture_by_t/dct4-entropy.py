# 엔트로피란?
# 엔트로피는 **"무질서도"**를 나타내는 숫자

import numpy as np
# 빨간 공 2개, 파란 공 3개
p_r = 2/5  # 빨간 공의 비율
p_b = 3/5  # 파란 공의 비율

# h_zero는 공이 섞여 있는 상태에서의 무질서도
h_zero = - p_r * np.log2(p_r) - p_b * np.log2(p_b)  # 엔트로피 계산
round(h_zero, 4)  # 결과: 0.971


# 빨간 공 1개, 파란 공 3개로 나눈 후의 엔트로피
p_r = 1/4  # 빨간 공의 비율
p_b = 3/4  # 파란 공의 비율
h_1_l = 0 # 빨간 공만 있는 쪽의 무질서도 0 
h_1_r = - p_r * np.log2(p_r) - p_b * np.log2(p_b)  # 파란 공쪽 엔트로피 계산
round(h_1_r, 4)  # 결과: 0.811


h_1 = (1/5 * h_1_l) + (4/5 *h_1_r) # 가중치
round(h_1, 4) # 0.649

# Information Gain
# 무질서도의 차이 
# 얼마나 깔끔하게 정리했는지 정도
# IG 값이 크다: 데이터를 나눈 후 무질서도가 많이 줄어들었다
# IG 값이 작다: 데이터를 나눈 후에도 무질서도가 별로 줄어들지 않았다
# 나눴을때 엔트로피가 더 높아지면 더이상 나누지 않는다

IG = h_zero - h_1
IG 

# ======================

# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수: bill_length_mm

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import OneHotEncoder

penguins = load_penguins()
penguins=penguins.dropna()

df_X=penguins.drop("species", axis=1)
df_X=df_X[["bill_length_mm", "bill_depth_mm"]]
y=penguins[['species']]


# 모델 생성
from sklearn.tree import DecisionTreeClassifier

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42)

param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

grid_search.fit(df_X,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

model = DecisionTreeClassifier(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(df_X,y)

from sklearn import tree
tree.plot_tree(model)

# value = 평균을 의미








