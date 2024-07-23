import pandas as pd
import numpy as np 

df = pd.DataFrame({"sex" : ["M","F", np.nan,"M","F"],
                   "score" : [5,4,5,4,np.nan]})
df 

#결측치 확인 

df["score"]+1
pd.isna(df)
pd.isna(df).sum()

#결측치 제거 
df.dropna() #모든 변수 결측치 제거 
df.dropna(subset = "score") #score 변수에서 결측치 제거 
df.dropna(subset = ["score", "sex"]) # 여러 변수 결측치 제거 

#데이터 프레임 location을 사용한 인덱싱 
#exam.loc[행 인덱스, 열 인덱스]
#iloc으로 활용하면 숫자 가능 

exam.loc[0,0]
exam.loc[[o],["id","nclass"]]

exam.iloc[0:2, 0:4] 

#결측치 대체
exam = pd.read_csv("data/exam.csv")
exam
exam.loc[[2,7,14], ["math"]] = np.nan
exam.loc[[2,7,14], ["math"]] = 3
exam

exam.iloc[[2,7,14], 2] = np.nan
exam

#수학점수 50 이하인 학생들 점수 다 50으로 상향조정
exam = pd.read_csv("data/exam.csv")
exam
exam.loc[exam["math"]<=50, "math"] = 50
exam

#영어어점수 90 이상인 학생들 점수 다 90으로 하향조정
#iloc 조회는 안됨
exam.loc[exam["english"]>=90, "english"] 
exam.iloc[exam["english"]>=90, 3] = 90

#iloc울 사용해서 조회하려면 무조건 숫자 백터가 들어가야함함
exam.iloc[np.array(exam["english"] >= 90), 3]
exam.iloc[np.where(exam["english"] >= 90[0]), 3]
exam.iloc[exam(exam["english"] >= 90]).index, 3]

exam

#math점수 50이하 "-"로 변경 
#경고- 문자로 바뀌니까 주의해라 
exam.loc[exam["math"]<=50, "math"] = "-"
exam


#"-"부분 결측치를 수학점수 평균으로 바꾸고 싶은 경우
#선생님 깃허브 참고
math_mean = exam.loc[(exam["math"] !="-"), "math"].mean()
exam.loc[(exam["math"] !="-"), "math"] = math.mean

exam.loc[exam["math"] ="-", "math"] = exam.query('math not in ["-"])['math']


exam["math"] = np.where(exam["math"] = "-", math_mean, exam["math"]

vector = np.array([np.nan if x == '=' else float(x) for x in exam["math"]])
vector = np.array([floatif x == '=' else float(x) for x in exam["math"]])

#6
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam["math"]= exam["math"].replace("-",math_mean)
exam



