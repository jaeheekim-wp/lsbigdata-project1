# 의사결정나무 함수 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 랜덤 시드 고정 (재현성을 위해)
np.random.seed(42)

# x 값 생성: -10에서 10 사이의 100개 값
x = np.linspace(-10, 10, 100)

# y 값 생성: y = x^2 + 정규분포 노이즈
y = x ** 2 + np.random.normal(0, 10, size=x.shape)

# 데이터프레임 생성
df = pd.DataFrame({'x': x, 'y': y})
df

# 데이터 시각화
plt.scatter(df['x'], df['y'], label='Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy Quadratic Data')
plt.legend()
plt.show()

# 입력 변수와 출력 변수 분리
X = df[['x']]  # 독립 변수는 2차원 형태로 제공되어야 함
y = df['y']

# 학습 데이터와 테스트 데이터로 분할 (80% 학습, 20% 테스트)
# 모델이 학습 데이터에만 적합하도록 과적합(overfitting)되는 것을 방지
# 새로운 데이터에 대한 예측 성능을 평가
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


# 디시전 트리 회귀 모델 생성 및 학습
model = DecisionTreeRegressor(random_state=42,
                              max_depth=6,
                              min_samples_split=10)
model.fit(X_train, y_train)


df_x=pd.DataFrame({"x": x})

# -10, 10까지 데이터에 대한 예측
y_pred = model.predict(df_x)
plt.scatter(df['x'], df['y'], label='Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Noisy Quadratic Data')
plt.legend()
plt.scatter(df_x['x'], y_pred, color="red")



# 모델 평가
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 테스트 데이터의 실제 값과 예측 값 시각화
plt.scatter(X_test, y_test, color='blue', label='Actual Values')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()

# R²(결정계수, R-squared)는 회귀 모델의 성능을 평가하는 지표
# 1에 가까울수록 모델이 데이터를 잘 설명
# R² 값이 0.8이라면, 모델이 y의 변동 중 80%를 설명하고 있다는 뜻