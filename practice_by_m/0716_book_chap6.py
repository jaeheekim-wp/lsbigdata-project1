import pandas as pd

#데이터 전처리 함수 
#query() 조건에 맞는 행을 걸러내는 
#df[]
#sort_values()
#groupby()
#

exam = pd.read_csv("data/exam.csv")
exam
exam.query("nclass==1")
exam.query("nclass!=1")
exam[exam["nclass"]==1] # 전에 


exam.query('math > 50')
exam.query('math < 50')
exam.query('english >= 50')
exam.query('english <= 80')
#and 
exam.query('nclass == 1 & math >=50')
exam.query('nclass == 2 and english >=80')

#or
exam.query('math >= 90 | english >=80')
exam.query('math >= 90 or english >=80')

exam.query('nclass not in [1,2]')
exam.query('nclass in [1,2]')


#exam[~exam["nclass"].isin([1,2])]

exam[["id","nclass"]]
exam[["nclass"]] 
#데이터 프레임 유지 
exam["nclass"] 


exam[["id","nclass"]] >3

#제거 
exam.drop(columns = "math")
exam.drop(columns = ["math","english"])


exam.query('nclass==1')[["math","english"]]
exam.query('nclass==1')\
            [["math","english"]]\
            .head()
            
            
#정렬
exam.sort_values("math")
exam.sort_values("math",ascending = False)
exam.sort_values(["nclass","english"], ascending = [True,False])
            
#변수추가
exam = exam.assign(
    total = exam["math"]+exam["english"]+exam["science"],
    mean = (exam["math"]+exam["english"]+exam["science"])/3)
    .sort_values("total", ascending = False)
exam.head()


#요약
#agg()
exam.agg(mean_math = ("math","mean"))
exam.groupby("nclass")\
    .agg(mean_math = ("math","mean"))


exam.groupby('nclass')\
    .agg(mean_math = ('math','mean'),
        sum_math = ('math','sum'),
        median_math = ('math','median'),
        n          = ('nclass','count'))
    


import pydataset
df= pydataset.data("mpg")
df

153p

mpg=pd.read_csv("data/mpg.csv")
mpg
mpg.query('manufacturer == "audi"')\
   .sort_values("hwy", ascending = False)
mpg.head()    

import pandas as pd


data = pd.read_csv("C:/Users/USER/Documents/LS 빅데이터스쿨/lsbigdata-project1/data/시도별_화재발생_현황_총괄__20240711173609_1.csv")
data
data_2020 = data[['행정구역별'] + data.filter(like='2020').columns.tolist()]
data_2020
data_2020.columns = data_2020.iloc[0]
data_2020 = data_2020[1:]
data_2020 = data_2020.reset_index(drop=True)
data_2020

data_2020_pop = data_2020.iloc[:, 0:4]
data_2020_pop
data_2020_pop['사망 (명)'] = pd.to_numeric(data_2020_pop['사망 (명)'])
data_2020_pop['부상 (명)'] = pd.to_numeric(data_2020_pop['부상 (명)'])
data_2020_pop['건수 (건)'] = pd.to_numeric(data_2020_pop['건수 (건)'])
data_2020_pop
data_2020_pop.info()

#변수이름 변경
data_2020_pop = data_2020_pop.rename(columns={"행정구역별":"area",
                                            "건수 (건)":"number",
                                            "사망 (명)":"death",
                                            "부상 (명)":"wound"})
data_2020_pop

fire.info()
fire.describe()
#새로운 변수 생성 
#사망률
fire = data_2020_pop.copy()
fire['ratio']=(fire["death"]/fire["number"])*100
fire
#인명피해
fire["total"] = fire["death"] + fire["wound"]
fire.head()

count_mean =  38659/17
count_mean

import numpy as np 

fire['dangerous'] = np.where(fire["number"]>=2274,"danger", "safe")
fire
fire['dangerous'] = np.where(fire["number"]>=count_mean,"danger", "safe")
fire

f,ire.head()

#빈도확인
import matplotlib.pyplot as plt
fire['dangerous'].value_counts()
fire_test = fire['dangerous'].value_counts()
fire_test.plot.bar(rot=0)
plt.show()


#행 필터
fire.query("ratio>=0.5")
fire.query("number>=1000")

fire.query
# 그룹 변수 기준으로 요약했는지 
# 묶을 수 있는 그룹이 ㅣ있는 데이터에서 필요 
fire.groupby("area")\
    .agg(mean_death = ("death","mean"),
         mean_wound = ("wound","mean"),
         sum_number = ("number","sum"))
         
         
#정렬했는지
fire.sort_values('number', ascending=False)
fire.sort_values(['number',"ratio"] , ascending=[True,False])



#데이터 합치기 
import pandas as pd
test1 = pd.DataFrame({"id"  :  [1,2,3,4,5],
                      "midterm" : [60,80,70,90,85 ]})
                      
test1                     
                      
                      
test2 = pd.DataFrame({"id"  :  [1,2,3,4,5],
                      "final" : [70,83,65,95,80]})

test2

test1 = pd.DataFrame({"id"  :  [1,2,3,4,5],
                      "midterm" : [60,80,70,90,85 ]})
                     
                      
test2 = pd.DataFrame({"id"  :  [1,2,3,40,5],
                      "final" : [70,83,65,95,80]})

test1                     
test2

#join방식 - 가로로         
total = pd.merge(test1, test2, how="left", on = "id")
total = pd.merge(test1, test2, how="right", on = "id")
total = pd.merge(test1, test2, how="inner", on = "id") #교집합
total = pd.merge(test1, test2, how="outer", on = "id") #합집합 
total

#exam에 name-teacher변수 추가 
name = pd.DataFrame({"nclass" : [1,2,3,4,5],
                     "teacher" : ["kim","lee","park","choi","jung"]})
name

exam = pd.read_csv("data/exam.csv")
exam

exam_new = pd.merge(exam, name, how="left", on = "nclass")
exam_new

#join방식 - 세로  ( 데이터 변수명이 같아야 함"score")

score1 = pd.DataFrame({"id"  :  [1,2,3,4,5],
                      "score" : [60,80,70,90,85 ]})
                     
                      
score2 = pd.DataFrame({"id"  :  [6,7,8,9,10],
                      "score" : [70,83,65,95,80]})

score1
score2

score_all = pd.concat([score1, score2])
score_all


test_all = pd.concat([test1, test2], axis=1) # 달라도 그냥 붙이고 싶을때 
test_all

