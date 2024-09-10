import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3) # 이상향 + 현실노이즈 

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df 

# train 만들기- 훈련 데이터 
train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

# valid 만들기
valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

# 라쏘 회귀 
from sklearn.linear_model import Lasso

val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha = i * 0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result

import seaborn as sns

df = pd.DataFrame({
    'l': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='l', y='tr')
sns.scatterplot(data=df, x='l', y='val', color='red')
plt.xlim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result) # 가장 작은 값의 인덱스를 반환 3>>네번째 
np.arange(0, 1, 0.01)[np.argmin(val_result)]
# np.arange(0, 1, 0.01)[3]

# ===================================
# 추정된 라쏘모델을 활용해서 그래프 그리기

model= Lasso(alpha=0.03)
model.fit(train_x,train_y)

#간격 0.01 에 대한 예측값 계산
k = np.arange(-4,4,0.01)

df2 = pd.DataFrame({
    "x" : k
})
df2

for i in range(2, 21):
    df2[f"x{i}"] = df2["x"] ** i

df2_y = model.predict(df2)

# valid set , valid set 에 대한 y
# expect_y = model.predict(valid_x)
plt.plot(df2["x"],df2_y,color="red")
plt.scatter(valid_x["x"],valid_y)
