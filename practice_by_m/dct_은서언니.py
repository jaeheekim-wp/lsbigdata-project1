# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘리스틱 넷 & 디시전트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수 : bill_length_mm


from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split


df = load_penguins()
penguins = df.dropna()

df.columns
# 'species', 'island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex', 'year'
df.info()  # 'species', 'island', 'sex' : 범주

penguins_dummies = pd.get_dummies(
                        penguins, 
                        columns=['species', 'island', 'sex'],
                        drop_first=True)

penguins_dummies.columns

x = penguins_dummies.drop(columns=('bill_length_mm'))
y = penguins_dummies['bill_length_mm']

len(penguins_dummies.columns)
len(x.columns)


train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2, random_state=42)


np.random.seed(20240905)

model = ElasticNet()  # alpha: 람다, l1_ratio : 알파

param_grid = {    # 하이퍼 파라미터 후보군들
    'alpha' : np.arange(0, 5 , 0.01),     
    'l1_ratio' : np.arange(0 , 1 , 0.01)
}


grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)


grid_search.best_params_ 
grid_search.cv_results_
result = pd.DataFrame(grid_search.cv_results_)
grid_search.best_score_    % 5.122
# alpha(0) -> 람다 패널티 0 일반 선형 회귀분석 , l1_ratio(0) -> alpha=0 릿지 회귀분석  => 결론 일반 선형 회귀분석

best_model = grid_search.best_estimator_ 
pred_y1 = best_model.predict(test_x)  # 바로 최적의 모델로 예측할 수 있음.

print("elasticnet test MSE :",np.mean((pred_y1 - test_y)**2))  # 5.642



# ------------------------------

from sklearn.tree import DecisionTreeRegressor

np.random.seed(20240905)
model2 = DecisionTreeRegressor()  # alpha: 람다, l1_ratio : 알파

param_grid = {    # 하이퍼 파라미터 후보군들
    'max_depth' : np.arange(0, 7),     
    'min_samples_split' : np.arange(2, 5 )
}


grid_search = GridSearchCV(
    estimator = model2,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)


grid_search.best_params_ 
grid_search.cv_results_
result = pd.DataFrame(grid_search.cv_results_)
grid_search.best_score_  # 8.169

# 최적의 하이퍼파라미터 : max_depth 6, min_samples_split 2

best_model2 = grid_search.best_estimator_  
pred_y2 = best_model2.predict(test_x)  


print( "의사결정나무 test MSE :",np.mean((pred_y2 - test_y)**2))  # 7.824