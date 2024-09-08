# 의사결정트리
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df=penguins.dropna()
df=df[["bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'bill_length_mm': 'y',
                   'bill_depth_mm': 'x'})

# x=15 기준으로 나눴을 때, 데이터 포인트가 몇개씩 나눠지는지
n1=df.query("x < 15").shape[0] # sum(df["x"]<15) # 1번 그룹
n2=df.query("x >= 15").shape[0] # sum(df["x"]>=15) # 2번 그룹
# pd.value_counts(df["x"]>=15)

# 1번 그룹, 2번 그룹 예측
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]

# 각 그룹 MSE
mse1=np.mean((df.query("x < 15")["y"]-y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"]-y_hat2)**2)

# x=15의 MSE 가중평균(데이터수-중요도)
# (mse1+mse2)/2
((mse1*n1)+(mse2*n2))/(len(df))

#=================================================
# x=20의 MSE 가중평균
n1=df.query("x < 20").shape[0] 
n2=df.query("x >= 20").shape[0]
y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]
mse1=np.mean((df.query("x < 20")["y"]-y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"]-y_hat2)**2)
((mse1*n1)+(mse2*n2))/(len(df))

#================================================
# 원래 MSE
np.mean((df["y"]-df["y"].mean())**2)

#================================================
# 기준값 x를 넣으면 MSE값이 나오는 함수
def my_mse(x):
    n1=df.query(f"x<{x}").shape[0] 
    n2=df.query(f"x>={x}").shape[0]
    y_hat1=df.query(f"x<{x}").mean()[0]
    y_hat2=df.query(f"x>={x}").mean()[0]
    mse1=np.mean((df.query(f"x<{x}")["y"]-y_hat1)**2)
    mse2=np.mean((df.query(f"x>={x}")["y"]-y_hat2)**2)
    return ((mse1*n1)+(mse2*n2))/(len(df))
my_mse(20)

df["x"].min() #13.1
df["x"].max() #21.5

# 13~22 사이 값 중 0.01 간격으로 MSE 계산
# minimize 사용하여 가장 작은 MSE가 나오는 x 값 찾기
x_values=np.arange(13.2, 21.4, 0.01)
len(x_values)
result=np.repeat(0.0, 820)
x_values[2]

for i in range(820):
    result[i] = my_mse(x_values[i])
result
np.argmin(result) #312번째 
x_values[np.argmin(result)] 

# np.argmin:주어진 배열에서 가장 작은 값의 인덱스(위치)를 반환하는 함수
# 깊이 1 
# 16.40999 
# ==============================

# 두번째 나눌때 기준값을 얼마가 되어야 하는지
# 깊이 2 트리의 기준값 두개
group1 = df.query("x < 16.41")  # 1번 그룹
group2 = df.query("x >= 16.41")  # 2번 그룹

def my_mse(data, x):
    n1=data.query(f"x<{x}").shape[0] 
    n2=data.query(f"x>={x}").shape[0]
    y_hat1=data.query(f"x<{x}").mean()[0]
    y_hat2=data.query(f"x>={x}").mean()[0]
    mse1=np.mean((data.query(f"x<{x}")["y"]-y_hat1)**2)
    mse2=np.mean((data.query(f"x>={x}")["y"]-y_hat2)**2)
    return ((mse1*n1)+(mse2*n2))/(n1+n2)

# 깊이 2의 첫번째 기준값
x_values1 = np.arange(group1['x'].min()+0.01, group1['x'].max(), 0.01)
result1 = np.repeat(0.0, len(x_values1))
for i in range(0, len(x_values1)):
    result1[i] = my_mse(group1, x_values1[i])
x_values1[np.argmin(result1)] # 14.01

# 깊이 2의 두번째 기준값
x_values2 = np.arange(group2['x'].min() + 0.01, group2['x'].max(), 0.01)
result2 = np.repeat(0.0, len(x_values2))
for i in range(0, len(x_values2)):
    result2[i] = my_mse(group2, x_values2[i])
x_values2[np.argmin(result2)] # 19.4

# ==============================================
# x,y 산점도를 그리고 빨간 평행선 4개 그리기
df.plot(kind="scatter", x="x", y="y")
# 세 개의 기준값(threshold)을 설정
threshold=[14.01, 16.42, 19.4]
# 기준값을 기준으로 데이터를 그룹으로 나누는 작업
df["group"]=np.digitize(df["x"], threshold)
df["group"].value_counts()
y_mean = df.groupby("group").mean()["y"]

k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.41, 100)
k3=np.linspace(16.41, 19.4, 100)
k4=np.linspace(19.4, 22, 100)

import matplotlib.pyplot as plt
plt.axvline(x=16.41, color='green', linestyle=':')
plt.axvline(x=14.01, color='green', linestyle=':')
plt.axvline(x=19.4, color='green', linestyle=':')
plt.plot(k1, np.repeat(y_mean[0], 100), color="red")
plt.plot(k2, np.repeat(y_mean[1], 100), color="red")
plt.plot(k3, np.repeat(y_mean[2], 100), color="red")
plt.plot(k4, np.repeat(y_mean[3], 100), color="red")