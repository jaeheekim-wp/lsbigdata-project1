<<<<<<< HEAD
# !pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
=======
import pandas as pd
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a

mpg=pd.read_csv("data/mpg.csv")
mpg.shape

<<<<<<< HEAD
sns.barplot(data = mpg, y = "hwy")

# 산점도 만들기 - 변수 간 관계표현 199p
# sns.scatterplot ( data = , x = "", y = "")
# sns.set( xlim = [ ],ylim = [ ]) # 축 범위 제한 
# 

plt.figure(figsize=(3, 2)) 
# 사이즈 조정
=======
# !pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize=(3, 2)) # 사이즈 조정
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
sns.scatterplot(data=mpg, 
                x="displ", y="hwy",
                hue="drv") \
   .set(xlim=[3, 6], ylim=[10, 30])
plt.show()
<<<<<<< HEAD
plt.clf()

# 그래프 설정 바꾸기기
plt.rcParams.update({'figure.dpi' : '150',
                     'figure.figsize' : [8,6],
                     'font.size' : '15',


# 설정 되돌리기
plt.rcdefaults
-----------------------------------------------------------------------

# 막대그래프 만들기 - 집단 간 차이 표현 205p

=======

# 막대그래프
>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a
# mpg["drv"].unique()
df_mpg=mpg.groupby("drv", as_index=False) \
          .agg(mean_hwy=('hwy', 'mean'))
df_mpg
<<<<<<< HEAD

plt.clf()


sns.barplot(data = df_mpg.sort_values("mean_hwy", ascending =  False ),
            x = "drv", y = "mean_hwy",
            hue = "drv")
plt.show()
plt.clf()

# as_index = False : 변수 인덱스화 하지 않고 유지 
# ascending =  False : 내림차순 ( 기본은 오름차순 )

# barplot : 행렬을 가진 가공된 데이터 
sns.barplot(data = df_mpg, x = "drv" , y = "n")
# countplot : 원 데이터를 바로 사용 
sns.countplot(data = mpg, x = "drv")


# 교재 8장, p.212
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

economics=pd.read_csv("./data/economics.csv")
economics.head()
economics.info()

sns.lineplot(data=economics, x="date", y="unemploy")
plt.show()
plt.clf()

# datetime 객체로 변환되면 날짜와 시간 데이터를 쉽게 조작하고 분석
economics["date2"]=pd.to_datetime(economics["date"])
economics
economics.info()

# dt 접근자를 사용하면 날짜 관련 속성을 쉽게 다룰 수 있음
economics[["date", "date2"]]
economics["date2"].dt.year # 어트리뷰트 
economics["date2"].dt.month
economics["date2"].dt.day
economics["date2"].dt.month_name() # 메서드
economics["date2"].dt.quarter
economics["quarter"]=economics["date2"].dt.quarter
economics[["date2", "quarter"]]
# 각 날짜는 무슨 요일인가?
economics["date2"].dt.day_name()
# 연산 
economics["date2"] + pd.DateOffset(days=30)
economics["date2"] + pd.DateOffset(months=1)
# 윤년 체크
economics["date2"].dt.is_leap_year 
# 변수 만들기 
economics["year"] = economics["date2"].dt.year
economics["year"]
economics

sns.lineplot(data=economics, 
             x='year', y='unemploy',
             errorbar=None)
             
# errorbar=None 옵션 : 신뢰구간 숨기기 
# 기본적으로 seaborn은 신뢰 구간을 표시하여 데이터의 불확실성을 시각화.            
             
sns.scatterplot(data=economics, 
             x='year', y='unemploy', s=2)
plt.show()
plt.clf()
economics.head(10)

my_df=economics.groupby("year", as_index=False) \
         .agg(
            mon_mean=("unemploy", "mean"),    # mean : 평균
            mon_std=("unemploy", "std"),      # std : 표준편차  
            mon_n=("unemploy", "count")
         )

my_df

# 신뢰구간(ci) 만들기 
mean + 1.96 * std/sqrt(12) 

my_df["left_ci"] = my_df["mon_mean"] - 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df["right_ci"] = my_df["mon_mean"] + 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df.head()

import matplotlib.pyplot as plt

x = my_df["year"]
y = my_df["mon_mean"]
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.scatter(x, my_df["left_ci"], color="blue", s=1)
plt.scatter(x, my_df["right_ci"], color="red", s=1)
plt.show()
plt.clf()

=======
plt.clf()
sns.barplot(data=df_mpg.sort_values("mean_hwy"),
            x = "drv", y = "mean_hwy",
            hue = "drv")
plt.show()

mpg


>>>>>>> 2a9062c51a0754a93a01446d84bf61ec8755369a

