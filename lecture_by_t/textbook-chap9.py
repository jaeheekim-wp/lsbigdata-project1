# !pip install pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

raw_welfare=pd.read_spss("./data/koweps/Koweps_hpwc14_2019_beta2.sav")
raw_welfare

welfare=raw_welfare.copy()
welfare.shape
welfare.describe()

# 변수명 변경 
welfare=welfare.rename(
    columns = {
        "h14_g3": "sex",
        "h14_g4": "birth",
        "h14_g10": "marriage_type",
        "h14_g11": "religion",
        "p1402_8aq1": "income",
        "h14_eco9": "code_job",
        "h14_reg7": "code_region"
    })


welfare=welfare[["sex", "birth", "marriage_type",
                "religion", "income", "code_job", "code_region"]]
welfare.shape

# 변수 검토
welfare["sex"].dtypes
welfare["sex"].value_counts()
# welfare["sex"].isna().sum()

# 성별 항목 이름 부여 
welfare["sex"] = np.where(welfare["sex"] == 1,'male', 'female')
# 빈도 구하기 
welfare["sex"].value_counts()

# income

welfare["income"].describe()
welfare["income"].value_counts()
welfare["income"].isna().sum()

# 성별 월급 평균표 구하기 
sex_income = welfare.dropna(subset="income") \
                  .groupby("sex", as_index=False) \
                  .agg(mean_income = ("income", "mean"))

sex_income

# 막대 그래프화 
sns.barplot(data=sex_income, x="sex", y="mean_income",
            hue="sex")
plt.show()
plt.clf()


# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산후 그리기
# 위 아래 검정색 막대기로 표시 - hw5로 공부하기 

#---------------------------------------
# # 9-3 235p
# 나이와 월급 상관관계 

# 나이 파생변수
welfare["birth"].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data=welfare, x="age")
plt.show()
plt.clf()


# age_income 나이에 따른 급여

age_income=welfare.dropna(subset="income") \
                    .groupby("age", as_index=False) \
                    .agg(mean_income = ("income", "mean"))

sns.lineplot(data=age_income, x="age", y="mean_income")
plt.show()
plt.clf()

# 나이별 "무응답자'구하기 by 샘 
# 나이별 income 칼럼 na 개수 세기!
welfare["income"].isna().sum()

welfare["income"].isna()
my_df=welfare.assign(income_na = welfare["income"].isna()) \
                        .groupby("age", as_index=False) \
                        .agg(n = ("income_na", "sum"))

sns.barplot(data = my_df, x="age", y="n")
plt.show()
plt.clf()

# 연령대별 월급 차이 240p

welfare["age"].head()
welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, "young",\
                                np.where(welfare['age'] <=59, "middle",\
                                                              "old")))

welfare["ageg"].value_counts()

sns.countplot(data = welfare, x = "ageg", hue = welfare["ageg"])
plt.show()
plt.clf()

# 연령대별 월급 평균표 만들기 
ageg_income = welfare.dropna(subset = 'income')\
                    .groupby('ageg',as_index = False)\
                    .agg(mean_income = ('income',"mean"))
ageg_income

sns.barplot(data = ageg_income, x = "ageg", y = "mean_income",
            order = ["young","middle","old"], hue = "ageg") # order 명령어로 정렬
plt.show()
plt.clf()


# 나이대별 수입 분석
# cut
# 각 연령 카테고리
bin_cut=np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])

welfare=welfare.assign(age_group = pd.cut(welfare["age"], 
                bins=bin_cut, 
                labels=(np.arange(12) * 10).astype(str) + "대"))

# np.version.version
# (np.arange(12) * 10).astype(str) + "대"

age_income = welfare.dropna(subset="income") \
                    .groupby("age_group", as_index=False) \
                    .agg(mean_income = ("income", "mean"))

age_income

sns.barplot(data=age_income, x="age_group", y="mean_income")
plt.show()
plt.clf()

-----
# 244p 연령대 및 성별 월급 차이 분석하기
# welfare["age_group"]의 dtype: category
welfare["age_group"]

welfare["age_group"] = welfare["age_group"].astype("object")
# 판다스 데이터 프레임을 다룰 때, 변수의 타입이
# 카테고리로 설정되어 있는 경우, groupby + agg 콤보가 안먹힘
# 그래서 object 타입으로 바꿔 준 후 수행

welfare["age_group"] = welfare["age_group"].astype("object")
sex_age_income = \
    welfare.dropna(subset="income")\
    .groupby(["age_group","sex"], as_index=False)\
    .agg(mean_income=("income","mean"))

sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="mean_income",
            hue="sex")
plt.show()
plt.clf()

# 연령대별, 성별 상위 4% 수입 찾아보세요!
# quantile 함수 설명

# x: 데이터가 저장된 배열. 이 배열에서 분위 값을 계산합니다.
# q=0.7: 찾고자 하는 분위 비율. 이 예에서는 70번째 백분위수에 해당하는 값을 찾습니다

x = np.arange(10)
np.quantile(x, q=0.7)
np.quantile(x, q=0.5)

welfare["age_group"] = welfare["age_group"].astype("object")

----------------------------------------------------------------------
# lambda x: 녀석을 어떻게 쓰는지에 대한 설명 예제 코드
# lambda x: 계산하려는 값을 지칭 - income

# vec- 변수
def my_f(vec):
    return vec.sum()

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: my_f(x)))
sex_age_income

-----------------------------------------------------------------------

# 진짜 ! 연령대별, 성별 상위 4% 수입 찾아보세요

sex_age_income = \
    welfare.dropna(subset="income")\
    .groupby(["age_group","sex"], as_index=False)\
    .agg(top4per_income=("income",
                          lambda x : np.quantile(x, q=0.96)))

sex_age_income


sns.barplot(data=sex_age_income,
            x="age_group", y="top4per_income",
            hue="sex")
plt.show()
plt.clf()


--------------------------------------------------------

## 참고
## 남규님 HW5 링크 
## 통계치를 한번에 구하려고 함수를 ㅇ선언해서 람다랑 쓰자는게 어제 강사님 요지였는데
## 함수 선언을 안하고도 agg에서 여러 통계치를 구하는 다른 방식인거지
sex_income2 = welfare.dropna(subset = 'income') \
                    .groupby('sex', as_index = False)[['income']] \
                    .agg(['mean', 'std'])


## 일준님님
sex_income3 = welfare.dropna(subset = 'income') \
                    .groupby('sex')[['income']] 
                    
---------------------------------------------------------------------------                    
                    
# 9-6장
# 직종 데이터 불러오기

welfare["code_job"]
welfare["code_job"].value_counts()
 
import pandas as pd
!pip install openpyxl
 
list_job = pd.read_excel("data/koweps/Koweps_Codebook_2019.xlsx", sheet_name = "직종코드")                  
 
welfare = welfare.merge(list_job,\
                         how="left", on = "code_job")                   
                    
                    
welfare.dropna(subset=["job", "income"])[["income","job"]]
 
 
job_income = welfare.dropna(subset=["job", "income"])\
                    .groupby("job", as_index = False)\
                    .agg(mean_income = ("income", "mean"))
                    
job_income.head()  

top10 = job_income.sort_values("mean_income", ascending = False).head(10)

top10

import matplotlib.pyplot as plt
plt.rcParams.update({"font.family" : "Malgun Gothic"})
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 7})
sns.barplot(data = top10, y = "job", x = "mean_income", hue = "job")
plt.show()
plt.clf()

--------------------------------------------------------------------------------

# 축 제목 글꼴 크기 설정
ax.set_xlabel("Mean Income", fontsize=14)
ax.set_ylabel("Job", fontsize=14)

# 축 눈금 글꼴 크기 설정
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# 범례 글꼴 크기 설정
plt.legend(fontsize=12)
plt.show()
plt.clf()

------------------------------------------------------------------------
# 9-7
# 성별에 따른  직업 빈도

job_male = welfare.dropna(subset = 'job') \
                  .query('sex == "male"') \
                  .groupby('job', as_index = False) \
                  .agg(n = ('job', 'count')) \
                  .sort_values('n', ascending = False) \
                  .head(10)

--------------------------------------------------------------------------
# 263p 9-8
# 종교 여부에 따른 이혼률 

welfare["marriage_type"]
rel_div = welfare.query("marriage_type != 5")\
                    .groupby("religion", as_index = False)\
                    ["marriage_type"]\
                    .value_counts(normalize=True) 
rel_div               
# count를 세주는 거에 normalize가 proportion(비율)을 세줌! 핵심 기억 


# 필터링
# round : 반올림 

rel_div = rel_div.query("marriage_type == 1")\
                 .assign(proportion = rel_div["proportion"]*100)\
                 .round(1)
rel_div


# round : 반올림 

--------------------------------------------------------------------------







