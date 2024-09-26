# 패키지 불러오기
import numpy as np
import pandas as pd

# adp 자료 61
# 표본 t‑test 함수 옵션 - 65p 
# t 검정  한꺼번에 진행하는 법 - 216p

# 대립가설의 형태에 따라서 alternative 옵션 조정 
# two-sided
# less
# greater

# 1:기본-1samp
# 2:그룹지을 수 있음 
# 3:짝을 지을 수 있나

tab3=pd.read_csv("./data/tab3.csv")
tab3

tab1=pd.DataFrame({"id": np.arange(1, 13),
                   "score": tab3['score']})
tab1

tab2=tab1.assign(gender=["female"]*7 + ["male"]*5)
tab2

# np.where 활용 
# tab2 = pd.DataFrame({
#     'id': np.arange(1, 13),
#     'score': tab3['score'],
#     'gender': np.where(np.arange(1, 13) < 8, 'female', 'male')
# })

# --------------------------------

# 1 
## 표본 t 검정 (그룹 1개)
## 귀무가설 vs. 대립가설
## H0: mu = 10 vs. Ha: mu != 10
## 유의수준 5%로 설정

from scipy.stats import ttest_1samp

result = ttest_1samp(tab1["score"], popmean=10, alternative='two-sided')
t_value=result[0] # t 검정통계량
p_value=result[1] # 유의확률 (p-value)
tab1["score"].mean() # 표본평균

result.statistic # t 검정통계량
result.pvalue # 유의확률 (p-value)
result.df # 자유도 

# 귀무가설이 참(mu=10)일 때, 11.53이 관찰될 확률이 6.48%이므로,
# 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인
# 0.05 (유의수준) 보다 크므로, 귀무가설을 거짓이라 판단하기 힘들다.
# 유의확률 0.0648이 유의수준 0.05보다 크므로
# 귀무가설을 기각하지 못한다.

 ## alpha > p_value : 귀무가설 기각
 ## alpha < p_value : 귀무가설 채택
 ## 기각역은 유의수준에 대응하는 x축의 영역

# 95% 신뢰구간 구하기
ci=result.confidence_interval(confidence_level=0.95)
ci[0]
ci[1]
## z0.975 z0.025 t0.975 t0.025

# ---------------------------------------------------------

# 2
## 표본 t 검정 (그룹 2) - 분산 같고, 다를때
## 분산 같은경우: 독립 2표본 t검정
## 분산 다를경우: 웰치스 t 검정
## 귀무가설 vs. 대립가설
## H0: mu_m = mu_f vs. Ha: mu_m > mu_f
## 유의수준 1%로 설정, 두 그룹 분산 같다고 가정한다.

from scipy.stats import ttest_ind


# 그룹을 나눠줘야함 
f_tab2=tab2[tab2["gender"] == "female"]
m_tab2=tab2[tab2["gender"] == "male"]


# alternative="less" 의 의미는 대립가설이
# 첫번째 입력그룹의 평균이 두번째 입력 그룹 평균보다 작다.
# 고 설정된 경우를 나타냄.

# # 기준은 항상 왼쪽.
result = ttest_ind(female["score"], male["score"],
                            alternative="less", equal_var=True) 
# result = ttest_ind(m_tab2["score"], f_tab2["score"], 
#                  alternative="greater", equal_var=True)

result.statistic
result.pvalue
result.df

# ---------------------------------------------------------

# 3
## 대응표본 t 검정 (짝지을 수 있는 표본)
## 귀무가설 vs. 대립가설
## H0: mu_before = mu_after vs. Ha: mu_after > mu_before
## H0: mu_d = 0 vs. Ha: mu_d > 0
## mu_d = mu_after - mu_before
## 유의수준 1%로 설정

# mu_d에 대응하는 표본으로 변환

# wide to long : pivot_table()
# pivot 활용: id열은 고정, group열 내 유니크한 value를 기준으로, score값을 채워줌

tab3
tab3_data = tab3.pivot_table(index='id', 
                             columns='group',
                             values='score').reset_index()
tab3_data                             

## mu_d = mu_after - mu_before
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

from scipy.stats import ttest_1samp
# 데이터 처리를 score_diff 하나로 정리했으니 1과 동일한 함수 사용 

result = ttest_1samp(test3_data["score_diff"], popmean=0, alternative='greater')
t_value=result[0] # t 검정통계량
p_value=result[1] # 유의확률 (p-value)
t_value; p_value


# long to wide : melt()
long_form = tab3_data.reset_index().melt(id_vars='id', \
                                         value_vars=['A', 'B'], \
                                         var_name='group', \
                                         value_name='score')

# 연습 1
df = pd.DataFrame({"id" : [1,2,3],\
                   "A" : [10,20,30],\
                   "B" : [40,50,60]})
                   
                   
df_long = df.melt(id_vars='id',\
                  value_vars=['A', 'B'],\
                  var_name='group',\
                  value_name='score')
                  
df_pivot = df_long.pivot_table(index = 'id',
                               columns = 'group',
                               values = 'score').reset_index()
 
 
# melt 함수는 데이터프레임을 '녹이는' 과정으로, 너비가 넓은 형태의 데이터를 길게 변환하는 데 사용됩니다. 
# 주로 여러 열이 하나의 변수로 나타내는 값을 가지는 경우에 사용됩니다. 
# melt 함수의 인수들을 설명하자면 다음과 같습니다:

# id_vars: 고정할 열을 지정합니다. 이 열들은 변환 과정에서 그대로 유지되며, 
# 나머지 열들이 길게 변환됩니다.
# value_vars: 녹일 열들을 지정합니다. 이 열들은 변수 이름과 값으로 변환됩니다.
# var_name: 변환 후 변수 이름을 저장할 새 열의 이름을 지정합니다.
# value_name: 변환 후 값들을 저장할 새 열의 이름을 지정합니다.

# pivot_table 함수는 길게 변환된 데이터프레임을 다시 넓은 형태로 변환하는 데 사용됩니다. 이를 통해 데이터의 특정 기준에 따라 집계된 결과를 얻을 수 있습니다. 각 인수의 의미를 설명하자면 다음과 같습니다:
# 
# index: 새로운 피벗 테이블에서 행 인덱스로 사용할 열을 지정합니다.
# columns: 새로운 피벗 테이블에서 열 레이블로 사용할 열을 지정합니다.
# values: 피벗 테이블에서 집계할 값을 가진 열을 지정합니다.

# 연습 1 - 결언니 4가지 
df = pd.DataFrame({
    "id" : [1,2,3],
    "A"  : [10,20,30],
    "B"  : [40,50,60]
})
df

df_long = df.melt(id_vars="id",
            value_vars=["A","B"],
            var_name="group",
            value_name="score")
df_long

df_wide_1 = df_long.pivot_table(index = "id",
                    columns = "group",
                    values = "score")
df_wide_1 # 기본
                    
df_wide_2 = df_long.pivot_table(
            columns = "group",
            values = "score")
df_wide_2

df_wide_3 = df_long.pivot_table(
            columns = "group",
            values = "score",
            aggfunc = "mean")
df_wide_3

df_wide_4 = df_long.pivot_table(
            columns = "group",
            values = "score",
            aggfunc = "max")
df_wide_4       

# # 연습 2       
# import seaborn as sns
# tips = sns.load_dataset("tips")
# tips 
# 
# 
# tips.pivot_table(
#     columns = "day",
#     values = "tip"
#     )
# 
# # 요일별로 펼치고 싶을때
# tips.reset_index(drop=False)\
#     .pivot_table(
#        index = tips.columns,
#        columns = 'day'
#        values = 'tip').reset_index() 
       
