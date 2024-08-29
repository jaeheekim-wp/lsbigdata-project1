import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({"x": x, "y": y})

# 다항식 항 추가 (2차부터 20차까지)
for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

# K-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)
alphas = np.arange(0, 1, 0.01)

# 결과를 저장할 배열 초기화
val_result = np.zeros(len(alphas))
tr_result = np.zeros(len(alphas))

# K-Fold 교차 검증 루프
for i, alpha in enumerate(alphas):
    fold_val_errors = []
    fold_tr_errors = []
    for train_index, valid_index in kf.split(df):
        train_df, valid_df = df.iloc[train_index], df.iloc[valid_index]
        train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
        train_y = train_df["y"]
        valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
        valid_y = valid_df["y"]

        # Lasso 모델 학습
        model = Lasso(alpha=alpha)
        model.fit(train_x, train_y)

        # 훈련 및 검증 데이터에 대한 예측
        y_hat_train = model.predict(train_x)
        y_hat_val = model.predict(valid_x)

        # 오차 계산
        fold_tr_errors.append(mean_squared_error(train_y, y_hat_train))
        fold_val_errors.append(mean_squared_error(valid_y, y_hat_val))

    # K-Fold 평균 오차 저장
    tr_result[i] = np.mean(fold_tr_errors)
    val_result[i] = np.mean(fold_val_errors)

# 성능 시각화
df_perf = pd.DataFrame({
    'l': alphas, 
    'tr': tr_result,
    'val': val_result
})

sns.scatterplot(data=df_perf, x='l', y='tr', label='Train Error')
sns.scatterplot(data=df_perf, x='l', y='val', color='red', label='Validation Error')
plt.xlim(0, 0.4)
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# 최적의 alpha 선택
best_alpha = alphas[np.argmin(val_result)]
print(f"Best alpha: {best_alpha}")

# 최적의 alpha로 최종 모델 학습
model = Lasso(alpha=best_alpha)
model.fit(df[["x"] + [f"x{i}" for i in range(2, 21)]], df["y"])

# 간격 0.01에 대한 예측값 계산
k = np.arange(-4, 4, 0.01)
df2 = pd.DataFrame({"x": k})
for i in range(2, 21):
    df2[f"x{i}"] = df2["x"] ** i
df2_y = model.predict(df2)

# 예측 결과 시각화
plt.plot(df2["x"], df2_y, color="red", label="Predicted")
plt.scatter(df["x"], df["y"], label="Original Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
