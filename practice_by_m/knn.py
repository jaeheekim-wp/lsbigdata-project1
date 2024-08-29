
# ================================

# Lasso

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# 데이터 로드(1번)
berry_train = pd.read_csv('data/blueberry/train.csv')
berry_test = pd.read_csv('data/blueberry/test.csv')
sub_df = pd.read_csv('data/blueberry/sample_submission.csv')

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
sub_df.to_csv("./data/blueberry/sample_submission_lasso.csv", index=False)

# ===============================

# Ridge 

# Ridge n_splits=20, np.arange(0.020, 0.025, 0.0001), model = Ridge(alpha=0.02499999999999997)


# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

# RMSE 계산 함수
def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, 
                                     cv=kf,
                                     n_jobs=-1,
                                     scoring="neg_mean_squared_error").mean())
    return score

# 각 alpha 값에 대한 교차 검증 점수 저장 (릿지에서 alpha는 lambda와 동일한 의미로 사용됩니다)
alpha_values = np.arange(0.006, 0.007, 0.00001)
mean_scores = np.zeros(len(alpha_values))

# 모델 평가
for i, alpha in enumerate(alpha_values):
    ridge = Ridge(alpha=alpha)
    mean_scores[i] = rmse(ridge)

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
plt.title('Ridge Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 최적의 릿지 회귀 모델 생성
model = Ridge(alpha=optimal_alpha)

# 모델 학습
model.fit(train_x, train_y)  # train 적용

# 테스트 데이터에 대한 예측
pred_y = model.predict(test_x)

# 예측 결과 저장
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/sample_submission_ridge.csv", index=False)


# =================================

# KNN

# 교차검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

# RMSE 계산 함수
def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, 
                                     cv=kf,
                                     n_jobs=-1,
                                     scoring="neg_mean_squared_error").mean())
    return score

# 각 n_neighbors 값에 대한 교차 검증 점수 저장
n_neighbors_values = np.arange(1, 21)  # 이웃 수를 1부터 20까지 시도
mean_scores = np.zeros(len(n_neighbors_values))

# KNN 모델 학습 및 평가
for i, n_neighbors in enumerate(n_neighbors_values):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    mean_scores[i] = rmse(knn)

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'n_neighbors': n_neighbors_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['n_neighbors'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('KNN Regression Train vs Validation Error')
plt.show()

# 최적의 n_neighbors 값 찾기
optimal_n_neighbors = df['n_neighbors'][np.argmin(df['validation_error'])]
print("Optimal n_neighbors:", optimal_n_neighbors)

# 최적의 KNN 모델 학습
model = KNeighborsRegressor(n_neighbors=optimal_n_neighbors)
model.fit(train_x, train_y)  # train 적용

# 테스트 데이터에 대한 예측
pred_y = model.predict(test_x)

# 예측 결과 저장
sub_df["yield"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/sample_submission_knn.csv", index=False)

# =========================

# 가중치 부여하기 
 
lasso = pd.read_csv("./data/blueberry/sample_submission_lasso.csv")
ridge = pd.read_csv("./data/blueberry/sample_submission_ridge.csv")
knn = pd.read_csv("./data/blueberry/sample_submission_knn.csv")


yield_ri = ridge["yield"]
yield_la = lasso["yield"]
yield_knn = knn["yield"]

yield_total = ((yield_ri * 7.5) + (yield_la * 1.5) + (yield_knn * 1))/10

sub_df["yield"] = yield_total

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/sample_submission_total.csv", index=False)


yield_total = ((yield_ri * 6) + (yield_la * 4))/10

sub_df["yield"] = yield_total

# csv 파일로 내보내기
sub_df.to_csv("./data/blueberry/submission_total3.csv", index=False)

