import pandas as pd

mpg = pd.read_csv("data/mpg.csv")
mpg.shape


import seaborn as sns
!pip install seaborn
import matplotlib.pyplot as plt

plt.figure(figsize = (3,2)) 
# 사이즈 조정 
sns.scatterplot(data = mpg,
                x = "displ",
                y = "hwy",
                hue = "drv")
   .set(xlim=[3,6],ylim =[10,30])
plt.show()
plt.clf()
                
# 막대그래프
mpg.groupby("drv") \
...     .agg(mean_hwy = ("hwy","mean"))

# 변수 두개인 데이터 프레임화 as_index
df_mpg = mpg.groupby("drv", as_index = False) \
            .agg(mean_hwy = ("hwy","mean"))

df_mpg.

sns.barplot(data = df_mpg.sort_values("mean_hwy", ascending = False),
            x = "drv", y = "mean_hwy", hue = "drv")
plt.show()

-----------------------------------------------
df_mpg = mpg.groupby("drv", as_index = False)\
            .agg(n = ("drv","count"))
            
df_mpg 

# barplot : 행렬을 가진 가공된 데이터 
sns.barplot(data = df_mpg, x = "drv" , y = "n")
# countplot : 원데이터를 바로 사용 
sns.countplot(data = mpg, x = "drv")


                
