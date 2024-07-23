import numpy as np  #과학 수치 계산
import pandas as pd # 데이터 가공 

# 열은 속성이자 변수 columnn
# 행은 로이자 케이스 row /case

df = pd.DataFrame({
    'name': ["김지훈", "이유진", "박동현", "김민지"],
    'english': [90, 80, 60, 70],
    'math': [50, 60, 100, 20]
})

# 데이터 추출 (열/변수)
df 
df["name"]  # 시리즈
df[["name"]] # 데이터프레임 

type(df)
type(df[["name"]])
type(df["name"])

# 평균
sum(df["english"])/4
df["english"].mean()

# df[("name", "english")]
df[["name", "english"]]

df["name"]

pd.show_versions()

# !pip install openpyxl
import pandas as pd

df_exam=pd.read_excel("data/excel_exam.xlsx")
df_exam

# 평균 구하기 
sum(df_exam["math"])/20
sum(df_exam["english"])/20
sum(df_exam["science"])/20
df_exam["math"].mean()
 
len(df_exam) # 함수
df_exam.shape # 메서드
df_exam.size # 메서드 

--------------------------------------------------------------------------------

# 변수 추가하기 

df_exam=pd.read_excel("data/excel_exam.xlsx", 
                      sheet_name="Sheet2")
df_exam   
df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"]
df_exam["mean"] = df_exam["total"] / 3
df_exam

# assign() 함수 활용해 추가 ( chap 6-157p) 
df_exam.assign(total = df_exam["math"] + df_exam["english"] + df_exam["science"],
               mean = lambda x : x["total"] / 3)
               
df_exam.assign(total =lambda x : x["math"] + x["english"] + x["science"],
               mean = lambda x : x["total"] / 3)

--------------------------------------------------------------------------------
# 데이터 추출 (열 안에 해당하는 조건의 행)
df_exam[df_exam["math"] > 50] 
df_exam[(df_exam["math"] > 50) & (df_exam["english"] > 50)] 

----------------------------------------------------------------------

# 평균 계산- numpy 활용 ( lec5 )
# 벡터 함수 사용하기 예제
# a = np.array([1, 2, 3, 4, 5])
# sum_a = np.sum(a) # 합계 계산
# mean_a = np.mean(a) # 평균 계산
# median_a = np.median(a) # 중앙값 계산
# std_a = np.std(a, ddof=1) # 표준편차 계산

mean_m = np.mean(df_exam["math"])
mean_e = np.mean(df_exam["english"]) # 39 평균구하기랑 비교 

df_exam[(df_exam["math"] > mean_m) & 
                (df_exam["english"] < mean_e)] 
                

df_nc3 = df_exam[df_exam["nclass"] == 3] # 3반인 데이터 행 추출 
df_nc3[["math", "english", "science"]] 
df_nc3[0:1]
df_nc3[1:2]
df_nc3[1:5]

df_exam[0:10:2]
df_exam[7:16]

-------------------------------------------------------------------------

#데이터 정렬하기 

#.sort_values() 오름차순 
# ascending = False 추가 시 내림차순

df_exam.sort_values("math", ascending=False) # 수학을 기준으로 내림차순 

df_exam.sort_values(["nclass", "math"], ascending=[True, False])
# nclass를 오름차순(true)으로 하고 그 안에서 math를 내림차순 (false)

# np.where 활용한 변수 추가 (chap5-121p)
np.where(a > 3, "Up", "Down")
df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down") 
df_exam["retest"] = np.where(df_exam["english"]>=75,"pass","retest")
df_exam
