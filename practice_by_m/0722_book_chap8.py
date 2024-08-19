import pandas as pd
<<<<<<< HEAD
import seaborn as sns
# !pip install seaborn
import matplotlib.pyplot as plt

=======
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a

mpg = pd.read_csv("data/mpg.csv")
mpg.shape


<<<<<<< HEAD
=======
import seaborn as sns
!pip install seaborn
import matplotlib.pyplot as plt

>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
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
<<<<<<< HEAD
# countplot : 원 데이터를 바로 사용 
sns.countplot(data = mpg, x = "drv")


# 교재 8장, p212

# 선 그래프
economics = pd.read_csv("data/economics.csv")
economics.head()
economics.info()

sns.lineplot(data =  economics, x = 'date', y = "unemploy")
plt.show()
plt.clf()
-----------------------------------------
# 0729
# 날짜 시간 타입 변수 만들기 
economics['date2'] = pd.to_datetime(economics['date'])
economics.info()

# 연,월,일 추출 
# 어트리뷰트 / 메서드 112p
economics['date2'].dt.year  # 어트리뷰트 
economics['date2'].dt.month  
economics['date2'].dt.month_name() # 메서드 
economics['date2'].dt.day
economics['date2'].dt.quarter
   
economics['quarter']= economics['date2'].dt.quarter 
economics[['date2', 'quarter']]               

# 각 날짜는 무슨 요일인가
economics['date2'].dt.day_name()

# 연산 
economics['date2'] + pd.DateOffset(days=3)
economics['date2'] + pd.DateOffset(days=30)
economics['date2'] + pd.DateOffset(months=1)
economics['date2'].dt.is_leap_year # 윤년 체크

economics['year'] = economics['date2'].dt.year 

sns.lineplot(data = economics, x = 'year', y = 'unemploy', errorbar = None)
sns.scatterplot(data = economics, x = 'year', y = 'unemploy', errorbar = None)
plt.show()
plt.clf()


my_df = economics.groupby('year', as_index = False)\
                 .agg(
                      mon_mean = ("unemploy", "mean"),
                      mon_std = ("unemploy", "std"),
                      mon_n = ("unemploy", "count")
                      )
# std : 표준편차 

my_df
mean + 1.96 * std/sqrt(12)
my_df["left_ci"] = my_df["mon_mean"] - 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df["right_ci"] = my_df["mon_mean"] + 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df.head()

x = my_df["year"]
y = my_df["mon_mean"]

plt.plot(x, y, color = "black")
plt.scatter(x, my_df["left_ci"], color='blue', zorder=10, s=2)
plt.scatter(x, my_df["right_ci"], color='red', zorder=10, s=2)
plt.show()
plt.clf()


=======
# countplot : 원데이터를 바로 사용 
sns.countplot(data = mpg, x = "drv")


                
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
