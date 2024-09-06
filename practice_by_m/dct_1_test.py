
# 의사결정트리
# f 스트링/ 리스트컴프리 이용!
# 지피티 코드에서 오류 잡아냄! 처음임! 감격스러움 

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df=penguins.dropna()
df=df[["bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'bill_length_mm': 'y',
                   'bill_depth_mm': 'x'})
df

# 원래 MSE는?
np.mean((df["y"] - df["y"].mean())**2)
29.81

# x=15 기준으로 나눴을때, 데이터포인트가 몇개씩 나뉘나요?
# 57, 276
n1=df.query("x < 15").shape[0]  # 1번 그룹
n2=df.query("x >= 15").shape[0] # 2번 그룹

# 1번 그룹은 얼마로 예측하나요?
# 2번 그룹은 얼마로 예측하나요?
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]

# 각 그룹 MSE는 얼마 인가요?
mse1=np.mean((df.query("x < 15")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"] - y_hat2)**2)

# x=15 의 MSE 가중평균은?
# (mse1 + mse2)*0.5 가 아닌
(mse1* n1 + mse2 * n2)/(n1+n2)
29.23
29.81 - 29.23

# x = 20일때 MSE 가중평균은?
n1=df.query("x < 20").shape[0]  # 1번 그룹
n2=df.query("x >= 20").shape[0] # 2번 그룹
y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]
mse1=np.mean((df.query("x < 20")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"] - y_hat2)**2)
(mse1* n1 + mse2 * n2)/(n1+n2)
29.73
29.81-29.73

# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_mse(data, x):
    n1=data.query(f"x<{x}").shape[0] 
    n2=data.query(f"x>={x}").shape[0]
    y_hat1=data.query(f"x<{x}").mean()[0]
    y_hat2=data.query(f"x>={x}").mean()[0]
    mse1=np.mean((data.query(f"x<{x}")["y"]-y_hat1)**2)
    mse2=np.mean((data.query(f"x>={x}")["y"]-y_hat2)**2)
    return ((mse1*n1)+(mse2*n2))/(n1+n2)

df["x"].min()
df["x"].max()

# 깊이 2 트리의 기준값 두개

# 첫 번째 기준값으로 나누기
first_split = 16.41

# 그룹 1: x < 16.41
group1 = df.query(f"x < {first_split}")

# 그룹 2: x >= 16.40
group2 = df.query(f"x >= {first_split}")

# 그룹 1에서 최적의 기준값 찾기
x_values_group1 = np.arange(group1["x"].min()+0.01, group1["x"].max(), 0.01)
result_group1 = np.array([my_mse(group1, x) for x in x_values_group1])
second_split_group1 = x_values_group1[np.argmin(result_group1)]
second_split_group1 # 14.01

# 그룹 2에서 최적의 기준값 찾기
x_values_group2 = np.arange(group2["x"].min()+0.01, group2["x"].max(), 0.01)
result_group2 = np.array([my_mse(group2, x) for x in x_values_group2])
second_split_group2 = x_values_group2[np.argmin(result_group2)]
second_split_group2 # 19.4 

## dct1의 for문이랑 비교해보기 


