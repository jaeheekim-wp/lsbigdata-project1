import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 데이터 로드
berry_train = pd.read_csv("train.csv")
berry_test = pd.read_csv("test.csv")
sub_df = pd.read_csv("sample_submission.csv")

# 결측치 확인
berry_train.isna().sum()
berry_test.isna().sum()

# 독립 변수와 종속 변수 분리
train_x = berry_train.drop("yield", axis=1)
train_y = berry_train["yield"]

test_x = berry_test.drop("yield", axis=1, errors='ignore')

# Polynomial Features 적용 (2차항 예시)
poly = PolynomialFeatures(degree=2, include_bias=False)
train_x_poly = poly.fit_transform(train_x)
test_x_poly = poly.transform(test_x)

# 데이터 표준화
scaler = StandardScaler()
train_x_poly_scaled = scaler.fit_transform(train_x_poly)
test_x_poly_scaled = scaler.transform(test_x_poly)

# 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=2024)

# RMSE 계산 함수
def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x_poly_scaled, train_y, 
                                     cv=kf,
                                     n_jobs=-1,
                                     scoring="neg_mean_squared_error").mean())
    return score

# ===================== Lasso 모델 =========================

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.006, 0.007, 0.00001)
mean_scores = np.zeros(len(alpha_values))

for i, alpha in enumerate(alpha_values):
    lasso = Lasso(alpha=alpha)
    mean_scores[i] = rmse(lasso)

# 결과를 DataFrame으로 저장
df_lasso = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha_lasso = df_lasso['lambda'][np.argmin(df_lasso['validation_error'])]
print("Optimal Lasso alpha:", optimal_alpha_lasso)

# 최적의 라쏘 모델 학습
lasso_model = Lasso(alpha=optimal_alpha_lasso)
lasso_model.fit(train_x_poly_scaled, train_y)

# 테스트 데이터에 대한 예측
pred_y_lasso = lasso_model.predict(test_x_poly_scaled)

# ==================== Ridge 모델 =====================

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0.006, 0.007, 0.00001)
mean_scores = np.zeros(len(alpha_values))

for i, alpha in enumerate(alpha_values):
    ridge = Ridge(alpha=alpha)
    mean_scores[i] = rmse(ridge)

# 결과를 DataFrame으로 저장
df_ridge = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha_ridge = df_ridge['lambda'][np.argmin(df_ridge['validation_error'])]
print("Optimal Ridge alpha:", optimal_alpha_ridge)

# 최적의 릿지 모델 학습
ridge_model = Ridge(alpha=optimal_alpha_ridge)
ridge_model.fit(train_x_poly_scaled, train_y)

# 테스트 데이터에 대한 예측
pred_y_ridge = ridge_model.predict(test_x_poly_scaled)

# ===================== KNN 모델 =====================

# 각 n_neighbors 값에 대한 교차 검증 점수 저장
n_neighbors_values = np.arange(1, 21)  # 이웃 수를 1부터 20까지 시도
mean_scores = np.zeros(len(n_neighbors_values))

for i, n_neighbors in enumerate(n_neighbors_values):
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    mean_scores[i] = rmse(knn)

# 결과를 DataFrame으로 저장
df_knn = pd.DataFrame({
    'n_neighbors': n_neighbors_values,
    'validation_error': mean_scores
})

# 최적의 n_neighbors 값 찾기
optimal_n_neighbors = df_knn['n_neighbors'][np.argmin(df_knn['validation_error'])]
print("Optimal KNN n_neighbors:", optimal_n_neighbors)

# 최적의 KNN 모델 학습
knn_model = KNeighborsRegressor(n_neighbors=optimal_n_neighbors)
knn_model.fit(train_x_poly_scaled, train_y)

# 테스트 데이터에 대한 예측
pred_y_knn = knn_model.predict(test_x_poly_scaled)

# =================== 가중치 부여 앙상블 ====================

# 예측 결과 불러오기
yield_ridge = pred_y_ridge
yield_lasso = pred_y_lasso
yield_knn = pred_y_knn

# 가중 평균 계산
yield_total = ((yield_ridge * 7.5) + (yield_lasso * 1.5) + (yield_knn * 1)) / 10

# 최종 예측 결과 저장
sub_df["yield"] = yield_total

# CSV 파일로 내보내기
sub_df.to_csv("./data/blueberry/sample_submission_total.csv", index=False)
