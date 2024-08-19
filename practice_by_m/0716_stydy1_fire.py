import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 정리
df = pd.read_csv("C:/Users/USER/Documents/카카오톡 받은 파일/시도별_화재발생_현황_총괄__20240711173609_1.csv")
df
df.columns

# 2020년도만 뽑아오기
data_2020 = df[['행정구역별'] + df.filter(like='2020').columns.tolist()]
data_2020

data_2020.columns = data_2020.iloc[0] #0번째 행을 열로
data_2020

data_2020 = data_2020[1:]
data_2020 = data_2020.reset_index(drop=True)
data_2020

data_2020_pop = data_2020.iloc[:, 0:4]
data_2020_pop

data_2020_pop.columns

pop = data_2020_pop.copy()
pop

pop.info() # 데이터타입 확인

# 문자형을 숫자형으로 변환
pop['건수 (건)'] = pd.to_numeric(pop['건수 (건)'])
pop['사망 (명)'] = pd.to_numeric(pop['사망 (명)'])
pop['부상 (명)'] = pd.to_numeric(pop['부상 (명)'])

# 변수명 변경
pop = pop.rename(columns = {"건수 (건)" : "건수"})
pop = pop.rename(columns = {"사망 (명)" : "사망자수"})
pop = pop.rename(columns = {"부상 (명)" : "부상자수"})

# 건수별로 정렬 
pop.sort_values("건수", ascending = False)

# 인명 피해
pop["total"] = pop["사망자수"] + pop["부상자수"]
pop.head()

# 위험도 추가
count_mean = 38659/17  #평균
pop["위험도"] = np.where(pop["건수"] >= count_mean, "dan", "saf")
pop.head()

# 빈도 막대 그래프
pop["위험도"].value_counts().plot.bar(rot=0) #rot=0 : 축 이름 수평
plt.show()
plt.clf()

# 시도별 인명피해 그래프
pop["total"].plot.bar(rot = 0)
plt.show()

----------------------------------------- re
#계절별 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/USER/Documents/LS 빅데이터스쿨/lsbigdata-project1/data/발화요인에_대한_월별_화재발생현황.csv")
df
df.columns

#연도 나누기 
data_2020 = df[['항목'] + df.filter(like='2020').columns.tolist()]
data_2020

data_2021 = df[['항목'] + df.filter(like='2021').columns.tolist()]
data_2021

data_2022 = df[['항목'] + df.filter(like='2022').columns.tolist()]
data_2022

#제품 결함 변수 삭제 - 행/열 맞추기 
data_2022 = data_2022.drop(columns = "2022.11")
data_2022


# 0번째 행을 칼럼으로 ( 열로 )
data_2020.columns = data_2020.iloc[0] 
data_2020

data_2021.columns = data_2021.iloc[0] 
data_2021

data_2022.columns = data_2022.iloc[0] 
data_2022

# 합계부터 내용 가져오기 (0,1행 제거)
data_2020 = data_2020[2:]
data_2021 = data_2021[2:]
data_2022 = data_2022[2:]
data_2020 
data_2021 
data_2022

# 인덱스 재정렬하기 
data_2020 = data_2020.reset_index(drop=True)
data_2020
data_2021 = data_2021.reset_index(drop=True)
data_2021
data_2022 = data_2022.reset_index(drop=True)
data_2022 

# year 변수 추가
data_2020['year']=2020
data_2021['year']=2021
data_2022['year']=2022
data_2020
data_2021
data_2022

# 세로로 합치기
data = pd.concat([data_2020, data_2021, data_2022])
data

# for문 사용하여 전체 열을 int로 변환
data.info()
columns_to_convert = ['계', '전기적요인', '기계적요인', '화학적요인',
                      '가스누출', '교통사고', '부주의', '기타', '자연적요인',
                      '방화', '방화의심', '미상']

for column in columns_to_convert:
    data[column] = pd.to_numeric(data[column])
    
data.info()

# 계절 파생변수 추가
data['계절']=np.where(data['항목'].isin(['12월','1월','2월']),'winter',
             np.where(data['항목'].isin(['3월','4월','5월']),'spring',
             np.where(data['항목'].isin(['6월','7월','8월']),'summer','fall')))
data
# 계절 순서를 봄, 여름, 겨울, 가을로 설정
order = ['spring', 'summer', 'fall', 'winter']
data['계절'] = pd.Categorical(data['계절'], categories=order, ordered=True)
data = data.sort_values(['year', '계절'])
data

# 계절별 화재 발생횟수 데이터프레임 생성
season=data.groupby(['year','계절']).agg(계절별화재=('계','sum'))
season

#선그래프 생성
#시각화 패키지 seaborn 
import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(
    data=season,
    x='계절',
    y='계절별화재',
    hue='year',
    marker='o')

plt.title('Seasonal Fire Incidents')
plt.xlabel('Season')
plt.ylabel('Number of Incidents')

plt.show()
plt.clf()


