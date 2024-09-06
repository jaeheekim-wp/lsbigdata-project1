import numpy as np
#빨2 / 파3
p_r = 2/5
p_b = 3/5
h_zero = - p_r * np.log2(p_r) - p_b * np.log2(p_b)
h_zero
round(h_zero, 4) #0.971

#빨1 / 파3
p_r = 1/4
p_b = 3/4
h_1_r = - p_r * np.log2(p_r) - p_b * np.log2(p_b)
h_1_r
round(h_1_r, 4)

h_1_l = 0

h_1 = (1/5 * h_1_l) + (4/5 *h_1_r) # 가중치
round(h_1, 4) # 0.649

# Information Gain
# 무질서도의 차이 
# 얼마나 깔끔하게 정리했는지 정도
# 값이 클수록 난장판이 적게 수정됨
# 나눴을때 엔트로피가 더 높아지면 더이상 나누지 않는다 

IG = h_zero - h_1
IG 

# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수: bill_length_mm

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import OneHotEncoder

penguins = load_penguins()
penguins.head()

## Nan 채우기
quantitative = penguins.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna(penguins[col].mean(), inplace=True)
penguins[quant_selected].isna().sum()

## 범주형 채우기
qualitative = penguins.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    penguins[col].fillna(penguins[col].mode()[0], inplace=True)
penguins[qual_selected].isna().sum()

df = penguins
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)
df

x=df.drop("species", axis=1)
y=df[['species']]
x
y

# 모델 생성
from sklearn.tree import DecisionTreeClassifier

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
model = DecisionTreeClassifier(random_state=42)
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

grid_search.fit(x,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

model = DecisionTreeClassifier(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(x,y)

from sklearn import tree
tree.plot_tree(model)