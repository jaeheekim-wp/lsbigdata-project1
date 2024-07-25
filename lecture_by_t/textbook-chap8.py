import pandas as pd

mpg=pd.read_csv("data/mpg.csv")
mpg.shape

# !pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize=(3, 2)) # 사이즈 조정
sns.scatterplot(data=mpg, 
                x="displ", y="hwy",
                hue="drv") \
   .set(xlim=[3, 6], ylim=[10, 30])
plt.show()

# 막대그래프
# mpg["drv"].unique()
df_mpg=mpg.groupby("drv", as_index=False) \
          .agg(mean_hwy=('hwy', 'mean'))
df_mpg
plt.clf()
sns.barplot(data=df_mpg.sort_values("mean_hwy"),
            x = "drv", y = "mean_hwy",
            hue = "drv")
plt.show()

mpg



