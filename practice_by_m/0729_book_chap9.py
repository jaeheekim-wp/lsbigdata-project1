!pip install pyreadstat
!pip install numpy
!pip install pandas
!pip install seaborn
!pip install matplotlib

import numpy as np
import pandas as pd
import seaborn as sns

# spss : 큰 데이터 가져올

raw_welfare = pd.read_spss("Data/koweps/Koweps_hpwc14_2019_beta2.sav")
raw_welfare 

welfare =  raw_welfare.copy()
welfare 

# 변수명 변경 
welfare = welfare.rename(columns = {
        "h14_g3" : "sex",
        "h14_g4" : "birth",
        "h14_g10": "marriage_type",
        "h14_g11": "religion",
        "p1402_8aq1" : "income",
        "h14_eco9" : "code_job",
        "h14_reg7" : "code_region"})
welfare

welfare = welfare[["sex", "birth","marriage_type","religion","income","code_job","code_region"]]

# 변수 검토 
welfare['sex'].dtypes
welfare['sex'].value_counts()
# welfare['sex'].isna().sum() # 0


welfare['sex'] = np.where(welfare["sex"] == 1,'male','female')
welfare['sex']

# 월급 변수 
welfare["income"].describe()
welfare['income'].isna().sum()

# 
sum(welfare["income"] > 9998)



sex_income = welfare.dropna(subset = 'income')\
                    .groupby('sex', as_index = False)\
                    .agg(mean_income = ('income',"mean"))

sex_income

-------------------------------------------------------------
# 그래프

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(data = sex_income, x = "sex",y = "mean_income",\
            hue = "sex")
plt.show()
plt.clf()

# 숙제-위 그래프에서 각 성별 95% 신뢰구간 계산 후 그리기 
# 위아래 검정색 막대기로 표시 

# 신뢰구간 
norm.ppf(0.975, loc=0, scale=1) 


my_df=economics.groupby("year", as_index=False) \
         .agg(
            mon_mean=("unemploy", "mean"),
            mon_std=("unemploy", "std"),      # std : 분산 
            mon_n=("unemploy", "count")
         )
my_df
mean + 1.96*std/sqrt(12)
my_df["left_ci"]=my_df["mon_mean"] - 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df["right_ci"]=my_df["mon_mean"] + 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df.head()

import matplotlib.pyplot as plt

x = my_df["year"]
y = my_df["mon_mean"]
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.scatter(x, my_df["left_ci"], color="blue", s=1)
plt.scatter(x, my_df["right_ci"], color="blue", s=1)
plt.show()
plt.clf()



welfare["birth"].describe()
sns.histplot(data = welfare, x = "birth")
plt.show()
plt.clf()

welfare['income'].isna().sum()


# 나이 파생변수 추가하기
welfare = welfare.assign( age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data = welfare, x = "age")
plt.show()
plt.clf()


age_income = welfare.dropna(subset = 'income')\
                    .groupby('age',as_index = False)\
                    .agg(mean_income = ('income',"mean"))
age_income


sns.lineplot(data = age_income, x = "age",y = "mean_income")
plt.show() 
plt.clf()

# 나이별  income 칼럼 na 갯수 세기!(- 나이별 무응답자 )
welfare['income'].isna().sum()

my_df = welfare.assign(income_na = welfare["income"].isna())\
                    .groupby('age', as_index = False)\
                    .agg(n = ('income_na',"sum"))

my_df

sns.barplot(data = my_df, x = "age", y ="n")
plt.show()

plt.clf()

my_df$["{1:변수명}"].inna().sum

----------------------------------------------

# 연령대별 월급 차이 240p

welfare["age"].head()
welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, "young",\
                                np.where(welfare['age'] <=59, "middle",\
                                                              "old")))

welfare["ageg"].value_counts()

sns.countplot(data = welfare, x = "ageg", hue = welfare["ageg"])
plt.show()
plt.clf()


ageg_income = welfare.dropna(subset = 'income')\
                    .groupby('ageg',as_index = False)\
                    .agg(mean_income = ('income',"mean"))
ageg_income

sns.barplot(data = ageg_income, x = "ageg", y = "mean_income",
            order = ["young","middle","old"], hue = "ageg")
plt.show()
plt.clf()
    
---------------------------------------------- 
# 나이대별 수입 분석 
# np.where 은 복잡해짐

## cut 예문 -> 연령대별로 확인하기 위하여
vec_x = np.array([1, 7, 5, 4, 6, 3])
cut = pd.cut(vec_x, 3)
cut.describe()
------------------------------------------------------------------

# 한결언니
vec_x = np.random.randint(0, 100, 50)
age_max=119
bin_cut = [10 * i + 9 for i in np.arange(age_max//10 + 1)]
pd.cut(vec_x, bins = bin_cut)

# 다경언니
vec_x = np.random.randint(0, 100, 50)
bin_cut = np.array([0:120:10])
pd.cut(welfare["age"], bin_cut)

# 현주
welfare['age_group'] = pd.cut(welfare['age'],
                         bins=[0, 9, 19, 29, 39, 49, 59, 69, np.inf], # 범위 나누기
                         labels=['baby', '10대', '20대', '30대', '40대', '50대', '60대','70대','80대', '90대','old'], # 이름 붙이기
                         right=False)

------------------------------------------------------------------

# 강사님

bin_cut = np.array([0, 9 , 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare = welfare.assign(age_group = pd.cut(welfare["age"], 
                        bins=bin_cut, 
                        labels = (np.arange(12)*10).astype(str)+"대"))
welfare["age_group"]         

ageg_income = welfare.dropna(subset = "income")\ 
                    .groupby("age_group", as_index = False)\
                    .agg(mean_income = ("income", "mean"))
sns.barplot(data = ageg_income, x="age_group", y="mean_income")
plt.show()
plt.clf()


np.version.version

-------------------------------------------------------------

# 244 
# 연령대 및 성별 월급차이

sex_income = \
    welfare.dropna(subset = "income")\
    .groupby(["ageg", "sex"], as_index = False)\
    .agg(mean_income = ("income", "mean"))

sex_income

---------------------------------------------------------------------------



# 연령대 별, 성별 상위 4 % 찾아보세요 
norm.ppf(0.95, loc=0, scale=1)





    
