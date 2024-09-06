import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y:펭귄 종류
# x1: 부리길이 bill_length_mm
# x2: 부리깊이 bill_depth_mm
df=penguins.dropna()
df=df[["species","bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'species' : 'y',
                      'bill_length_mm': 'x1',
                      'bill_depth_mm': 'x2'})
df

# x1,x2 산점도 그리기 , 점 색깔은 종별로 다르게 
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='x1', y='x2', hue='y')
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.title("Penguin Species by Bill Length and Depth")
# 수직선 추가 
plt.axvline(x = 45)
plt.show()

# Q1. 나누기 전 현재의 엔트로피?
# 입력값이 벡터 > 엔트로피!
p_i = df['y'].value_counts() / len(df['y'])
entropy_curr = -sum(p_i * np.log2(p_i))

# Q2. x = 45로 나눴을때, 엔트로피 평균?
n1 = df.query("x1 < 45").shape[0]  # 1번 그룹
n2 = df.query("x1 >= 45").shape[0] # 2번 그룹

# 1번 그룹은 어떤 종류로 예측?
# 2번 그룹은 어떤 종류로 예측?
y_hat1 = df.query("x1 < 45")['y'].mode()
y_hat2 = df.query("x1 >= 45")['y'].mode()

# 각 그룹 엔트로피?
p_1 = df.query("x1 < 45")['y'].value_counts() / len(df.query("x1 < 45")['y'])
entropy1 = -sum(p_1 * np.log2(p_1))

p_2 = df.query("x1 >= 45")['y'].value_counts() / len(df.query("x1 >= 45")['y'])
entropy2 = -sum(p_2 * np.log2(p_2))

entropy_x1_45 = (n1 * entropy1 + n2 * entropy2) / (n1 + n2)

# =======================

# 기준값 x를 넣으면 entropy값이 나오는 함수는?
def my_entropy(x): 
    # x기준으로 나눴을때, 데이터 포인터가 몇개 씩 나뉘나요?
    n1 = df.query(f"x1 < {x}").shape[0] # 1번 그룹
    n2 = df.query(f"x1 >= {x}").shape[0] # 2번 그룹
    # 각 그룹 엔트로피는 얼마 인가요?
    p_1 = df.query(f"x1 < {x}")['y'].value_counts() / len(df.query(f"x1 < {x}")['y'])
    entropy1 = -sum(p_1 * np.log2(p_1))
    p_2 = df.query(f"x1 >= {x}")['y'].value_counts() / len(df.query(f"x1 >= {x}")['y'])
    entropy2 = -sum(p_2 * np.log2(p_2))
    # 가중 평균 엔트로피 
    entropy_x = (n1*entropy1 + n2*entropy2) / (n1+n2)
    return(entropy_x)

my_entropy(45)

# x1 기준으로 최적 기준값은 얼마인가?

# entropy계산을 해서 가장 작은 entropy가 나오는 x는?
import numpy as np
from scipy import optimize

entropy_list = []
x_list = np.arange(df["x1"].min(), df["x1"].max(), 0.01)

for i in x_list : 
    entropy_list.append(my_entropy(i))

entropy_list

# entropy_list 최소값
min(entropy_list)

# entropy_list를 최소로 만드는 x의 값
x_list[np.argmin(entropy_list)]


# =================== 한렬 ========================
# 최적의 분할점 찾는 함수
def find_best_split(df, x_values):
    best_split = None
    min_entropy = float('inf')  # 매우 큰 값으로 초기화
    
    for x in x_values:
        entropy_x = my_entropy(x)  # my_entropy 함수 사용
        if entropy_x < min_entropy:
            min_entropy = entropy_x
            best_split = x
    
    return best_split, min_entropy

# x1(부리 길이)의 고유한 값들에 대해 최적 분할점 찾기
x_values = np.arange(df["x1"].min(), df["x1"].max(), 0.01)
best_split, min_entropy = find_best_split(df, x_values)

print(f"최적의 x1 분할점: {best_split}")
print(f"최소 엔트로피: {min_entropy}")

my_entropy(42.30999999999797)

# 최적 -> 42.31
# 그때의 엔트로피 -> 80.43
# =================== 한렬 ========================


# x, y 산점도를 그리고, 빨간 평행선 4개 그려주세요!
import matplotlib.pyplot as plt

df.plot(kind="scatter", x="x", y="y")
thresholds=[14.01, 16.42, 19.4]
df["group"]=np.digitize(df["x"], thresholds)
y_mean=df.groupby("group").mean()["y"]
k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k3=np.linspace(16.42, 19.4, 100)
k4=np.linspace(19.4, 22, 100)
plt.plot(k1, np.repeat(y_mean[0],100), color="red")
plt.plot(k2, np.repeat(y_mean[1],100), color="red")
plt.plot(k3, np.repeat(y_mean[2],100), color="red")
plt.plot(k4, np.repeat(y_mean[3],100), color="red")