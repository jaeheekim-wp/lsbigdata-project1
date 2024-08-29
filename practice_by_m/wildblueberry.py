import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


import os
# 워킹 디렉토리 확인 
os.getcwd()
# 워킹 디렉토리 변경 
os.chdir('c:/Users/USER/Documents/LS 빅데이터스쿨/lsbigdata-project1/data/blueberry')

# Regression with a Wild Blueberry Yield Dataset

berry_train = pd.read_csv("train.csv")
berry_test = pd.read_csv("test.csv")
sub_df = pd.read_csv("sample_submission.csv")

# 결측치 없음/ 범주형도 없음- 더미 진행 x 
# ==========================================
# berry_train에서 yield를 y로, 나머지를 x로 분리

berry_train_y = berry_train[["yield"]] # yield 열만 가져옴
berry_train_y
berry_train_x = berry_train.drop(columns=[["yield", "id"]]) # yield와 Id을 제외한 나머지 열들만 가져옴.
berry_train_x


# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(berry_train_x)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_scaled, berry_train_y, cv=kf,
                                     scoring="neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 10, 0.01)
mean_scores = np.zeros(len(alpha_values))

k = 0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 최적의 alpha 값으로 모델 학습
model = Lasso(alpha=optimal_alpha)
model.fit(berry_train_x, berry_train_y)

# berry_test_x 준비
berry_test_x = berry_test.drop(columns=["id"])

# train과 test에서 차이가 나는 열 확인
extra_columns_in_test = set(berry_test_x.columns) - set(berry_train_x.columns)
extra_columns_in_train = set(berry_train_x.columns) - set(berry_test_x.columns)

extra_columns_in_test, extra_columns_in_train

# 테스트 데이터에 대해 예측 수행
pred_y = model.predict(berry_test_x)

# 예측 결과를 제출 파일에 반영
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/sample_submission0828_3.csv", index=False)

# ========================================================================
# 2조(관세음보살)
# 팀원: 안상후, 윤대웅, 정아영, 백선아
# 스코어: 357.02855

# 필요한 데이터 불러오기
berry_train=pd.read_csv("./data/blueberry/train.csv")
berry_test=pd.read_csv("./data/blueberry/test.csv")
sub_df=pd.read_csv("./data/blueberry/sample_submission.csv")

berry_train.isna().sum()
berry_test.isna().sum()

berry_train.info()

## train
X=berry_train.drop(["yield", "id"], axis=1)
y=berry_train["yield"]
berry_test=berry_test.drop(["id"], axis=1)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled=scaler.transform(berry_test)

# 정규화된 데이터를 DataFrame으로 변환
X = pd.DataFrame(X_scaled, columns=X.columns)
test_X= pd.DataFrame(test_X_scaled, columns=berry_test.columns)

polynomial_transformer=PolynomialFeatures(3)

polynomial_features=polynomial_transformer.fit_transform(X.values)
features=polynomial_transformer.get_feature_names_out(X.columns)
X=pd.DataFrame(polynomial_features,columns=features)

polynomial_features=polynomial_transformer.fit_transform(test_X.values)
features=polynomial_transformer.get_feature_names_out(test_X.columns)
test_X=pd.DataFrame(polynomial_features,columns=features)

#######alpha
# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(2, 4, 1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()


### model
model= Lasso(alpha=2.9)

# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y=model.predict(test_X) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["yield"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/sample_submission_scaled.csv", index=False)

# =================================================

#조:3
#팀원:이재준,권효은,이태현,오현욱
#스코어(private): 359.15721

## 필요한 데이터 불러오기
house_train=pd.read_csv("data/blueberry/train.csv")
house_test=pd.read_csv("data/blueberry/test.csv")
sub_df=pd.read_csv("data/blueberry/sample_submission.csv")


house_train=house_train.iloc[:,1:]
house_test=house_test.iloc[:,1:]

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)

#df.info()

df.select_dtypes(include=[object]).columns

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## train
train_x=train_df.drop("yield",axis=1)
train_y=train_df["yield"]

## test
test_x=test_df.drop("yield",axis=1)

train_x.columns
for col in train_x.columns:
    train_x[f'{col}_pow_6'] = train_x[col] ** 6
test_x.columns
for col in test_x.columns:
    test_x[f'{col}_pow_6'] = test_x[col] ** 6

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024) #셔플한 값 만들기

# 알파 값 설정 처음에는 값간격을 크게하고 범위를 넓혀서 찾은후
# 세세한 값을 찾기 위해서 값간격을 작게하고 범위를 좁혀서 세세한 값을 찾는다
alpha_values = np.arange(0,1 , 0.1)

# 각 알파 값에 대한 교차 검증 점수 저장
mean_scores = np.zeros(len(alpha_values))

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf, n_jobs=-1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)


k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)


# 선형 회귀 모델 생성
model = Lasso(alpha=0)

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌


test_x.columns[test_x.isna().sum()>0]

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["yield"] = pred_y
sub_df
# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/all_Lasso_rain.csv", index=False)

# =====================================================

# 7조
# 이용규, 김재희, 송현주, 박수빈 
# 362.44

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

# 데이터 불러오기
train = pd.read_csv('../data/WildBlueberry/train.csv')
test = pd.read_csv('../data/WildBlueberry/test.csv')
sub = pd.read_csv('../data/WildBlueberry/sample_submission.csv')

# 데이터 전처리
train_n = len(train)
df = pd.concat([train, test], ignore_index=True)

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Train/Test 데이터셋 분리
train_df = df_imputed.iloc[:train_n, :]
test_df = df_imputed.iloc[train_n:, :]

# X와 y 변수 분리 (Train 데이터)
X = train_df.drop(columns=['yield'])
y = train_df['yield']

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 모델 초기화 - 최적의 alpha 값 사용
lasso_alpha = 1.41
ridge_alpha = 0.31

lr = LinearRegression()
lasso = Lasso(alpha=lasso_alpha)
ridge = Ridge(alpha=ridge_alpha)

# 교차 검증 예측 수행
kf = KFold(n_splits=10, shuffle=True, random_state=2024)
pred_lr = cross_val_predict(lr, X_scaled, y, cv=kf)
pred_lasso = cross_val_predict(lasso, X_scaled, y, cv=kf)
pred_ridge = cross_val_predict(ridge, X_scaled, y, cv=kf)

# 가중 평균을 사용하여 최적의 가중치를 찾는 함수 정의
def weighted_mae(weights):
    final_pred = weights[0] * pred_lr + weights[1] * pred_lasso + weights[2] * pred_ridge
    return mean_absolute_error(y, final_pred)

# 가중치 초기값 및 제약조건 설정 (모든 가중치의 합이 1)
initial_weights = [0.33, 0.33, 0.33]
constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
bounds = [(0, 1), (0, 1), (0, 1)]  # 각 가중치는 0에서 1 사이

# 최적화 수행
optimal_weights = minimize(weighted_mae, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = optimal_weights.x
print(f"Optimal weights: {best_weights}")

# 최적의 가중치로 최종 예측 수행
X_test = test_df.drop(columns=['yield'], errors='ignore')
X_test_scaled = scaler.transform(X_test)

predictions_lr = lr.fit(X_scaled, y).predict(X_test_scaled)
predictions_lasso = lasso.fit(X_scaled, y).predict(X_test_scaled)
predictions_ridge = ridge.fit(X_scaled, y).predict(X_test_scaled)

final_predictions = best_weights[0] * predictions_lr + best_weights[1] * predictions_lasso + best_weights[2] * predictions_ridge

# 결과 저장
# 최종 예측값을 yield 열에 삽입
sub['yield'] = final_predictions

# 파일로 저장
sub.to_csv('../data/WildBlueberry/sample_submission.csv', index=False)

# =================================================

# 조: 6조
# 팀원: 이승학, 박유나, 김연예진, 오서연
# 스코어: 362.43823
# 코드:

# 가중치 부여하기 
ridge = pd.read_csv("./data/blueberry/ridge.csv")
lasso = pd.read_csv("./data/blueberry/lasso.csv")
linear = pd.read_csv("./data/blueberry/linear.csv")


yield_ri = ridge["yield"]
yield_la = lasso["yield"]
yield_li = linear["yield"]

yield_total = ((yield_ri * 6) + (yield_la * 3) + (yield_li * 1))/10

sub_df["yield"] = yield_total

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/submission_total2.csv", index=False)


yield_total = ((yield_ri * 6) + (yield_la * 4))/10

sub_df["yield"] = yield_total

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/submission_total3.csv", index=False)

# ====================================

# Ridge 

# Ridge n_splits=20, np.arange(0.020, 0.025, 0.0001), model = Ridge(alpha=0.02499999999999997)

kf = KFold(n_splits=20, shuffle=True, random_state=2024)        def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, 
                                     cv = kf,
                                     n_jobs = -1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.020, 0.025, 0.0001)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Ridge Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

model = Ridge(alpha=0.02499999999999997) 

# 모델 학습
model.fit(train_x, train_y) # train 적용

pred_y=model.predict(test_x) # test로 predict 하기

# ================================

# Lasso

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 로드(1번)
berry_train = pd.read_csv('Blueberry/train.csv')
berry_test = pd.read_csv('Blueberry/test.csv')
sub_df = pd.read_csv('Blueberry/sample_submission.csv')

berry_train.isna().sum()
berry_test.isna().sum()

train_x = berry_train.drop("yield", axis = 1)
train_y = berry_train["yield"]

test_x = berry_test.drop("yield", axis=1, errors='ignore')

kf = KFold(n_splits=10, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, 
                                     cv = kf,
                                     n_jobs = -1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.006, 0.007, 0.00001)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 선형 회귀 모델 생성(8번)
model = Lasso(alpha=0.006769999999999969) # valid는 lambda 알기 위해서 쓰는 것

# 모델 학습
model.fit(train_x, train_y) # train 적용

pred_y=model.predict(test_x) # test로 predict 하기

# SalePrice 바꿔치기
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("submission/sample_submission_berry_1.csv", index=False)

# =======

# 선형회귀

from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 예측값 계산
train_x = berry_train.drop("yield", axis=1)
train_y = berry_train["yield"]

test_x = berry_test

pred_y = model.predict(test_x)

sub_df["yield"] = pred_y

sub_df.to_csv("sample_submission_linear.csv", index=False)


