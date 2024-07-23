import pandas as pd
import numpy as np

pd.DataFrame()
df= pd.DataFrame({
    '제품' : ["사과", "딸기","수박"],
    '가격' : [1800,1500,3000],
    '판매량' : [24,38,13]
    })
df
sum(df['가격'] / 3)

#-----86p------
import pandas as pd
df_exam=pd.read_excel("data/excel_exam.xlsx")

df_exam


sum(df_exam["math"])  /20
sum(df_exam["english"]) /20
sum(df_exam["science"]) /20

df_exam.shape
len(df_exam)
df_exam.size

df_exam=pd.read_excel("data/excel_exam.xlsx",sheet_name = "Sheet2")

df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"]
df_exam
df_exam["mean"] = df_exam["total"]/3
df_exam

df_exam["math"]>50
df_exam[df_exam["math"]>50]  # true에 해당대는 행들 추출
(df_exam["math"]>50) & (df_exam["english"]>50)

df_exam
mean_m=np.mean(df_exam["math"])
mean_e=np.mean(df_exam["english"])
df_exam[(df_exam["math"] > mean_m) &
            (df_exam["english"] < mean_e)]
df_exam

df_nc3=df_exam[df_exam["nclass"]==3]
df_nc3[["math","english","science"]]
df_nc3[1:4]


a= np.array([4,2,5,3,6])
a[2]

df_exam
df_exam[0:10]
df_exam[7:16]
df_exam[0:10:2]
# 행 기준으로 인덱싱 가능 

#mpg데이터 살피기 105p
mpg=pd.read_csv("data/mpg.csv")
mpg
mpg.head(10)
mpg.shape
mpg = mpg.rename(columns = {"cty":"city"})
mpg = mpg.rename(columns = {"hwy":"highway"})
mpg
#열(columns): 데이터 프레임의 수직선.
#행(rows): 데이터 프레임의 수평선 .

#파생변수 만들기116p
df= pd.DataFrame({"var1":[4,3,8],
                    "var2":[2,6,1]})
df
#합계 파생변수 
df['var_sum']=df["var1"]+df["var2"]
df
#평균 파생변수 
df['var_mean']=(df["var1"]+df["var2"]) /2 
df

#통합연비변수
mpg["total"]= (mpg["cty"] + mpg["hwy"]) / 2
mpg
mpg.head()
sum(mpg["total"]) / len(mpg) #함수활용
mpg["total"].mean() #메서드활용


#조건문 활용 파생변수 118p
#mpg["total"] 의 평균 mean / 중앙값 50% 확인 
mpg["total"].describe()

#히스토그램 
import matplotlib.pyplot as plt

mpg["total"].plot.hist()
plt.show()
plt.clf()

#합격 판정 변수
import numpy as np
mpg["test"]= np.where (mpg["total"]>=20,"pass", "fail")
mpg
mpg.head()

##빈도표 만들기 value_counts()
mpg["test"].value_counts()
count_test= mpg["test"].value_counts()
#막대그래프 
count_test.plot.bar(rot=0)
plt.show()

#연비 등급 변수 ( 중첩 조건문 활용 )
mpg["grade"]= np.where(mpg["total"]>=30,"A",
              np.where(mpg["total"]>=20,"B","C"))
mpg.head()
##빈도표/막대그래프
count_grade= mpg["grade"].value_counts()
count_grade.plot.bar(rot=0)
plot.show()

### 알파벳 순으로 막대 정렬
count_grade= mpg["grade"].value_counts().sort_index()
count_grade.plot.bar(rot=0)
plt.show()




#151P 오름차순/ 내림차순 
df_exam.sort_values("math",ascending= False)
df_exam.sort_values(["nclass","math"],ascending= [True,False])


np.where(a>3,"up","down")
df_exam["math"]>50
df_exam["updowm"]=np.where(df_exam["math"]>50,"Up","Down")
df_exam
