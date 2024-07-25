import pandas as pd
import numpy as np

# 데이터 전처리(data preprocessing) 
# pandas 패키지 함수

# query()  행 추출 
# df[] 열(변수)추출 ## ex) df["total"] 
# drop() 변수 버리기
# sort_values() 정렬 
# assign() 변수 추가
# groupby() 집단별로 나누기 
# agg() 통계치 구하기 
# merge() 가로로 데이터 합치기 (열)
# concat() 세로로 데이터 합치기 (행)

-----------------------------------------------------------------

exam = pd.read_csv("data/exam.csv")

# 행 걸러내기 .query()

# 조건에 맞는 행을 걸러내는 
# "해당 변수 언급 필수 "

# exam[exam["nclass"] == 1]

exam.query("nclass == 1")
exam.query("nclass != 1")
exam.query("nclass != [1,2]")
## and /& 둘다 충족 
exam.query("nclass == 1 & math > 50")
exam.query("nclass == 1 and math > 50")
## or| 하나라도 충족 
exam.query("nclass == 1 | nclass == 2")
exam.query("nclass == 1 or nclass == 2")
### 더 간단하게 
exam.query("nclass in [1, 2]")
### 반대로 (~활용)
exam[~exam["nclass"].isin([1,2])]

#문자 변수일 때: 큰 따옴표 안 작은 따옴표 or 작은 따옴표 안 큰 따옴표 
mpg = pd.read_csv("data/mpg.csv")
mpg
mpg.query("manufacturer == 'volkswagen'")
mpg.query('manufacturer == "audi"')

------------------------------------------------------------
 ## cf) chap 4 데이터 추출 방식 변화 
 ## exam[exam["math"] > 50] 
exam.query("math > 50")
 ## exam[(exam["math"] > 50) & (exam["english"] > 50)]
exam.query("math > 50 & english > 50")
------------------------------------------------------------

# 모든 행 출력하도록 설정 
pd.set_option(displat) # p.143

------------------------------------------------------------

# 변수(열) 꺼내오기 
exam["nclass"] #시리즈 
exam[["nclass"]] #데이터프레임 
exam[["id", "nclass"]]

----------------------------------------------------------

# 변수 버리기 .drop()
exam.drop(columns = ["math", "english"])
exam
 ## ch)exam2 = exam2.rename(columns={"nclass" : "class"}) - 중괄호
 
----------------------------------------------------------------

# 행 +열 같이 
# query() 와 [] 조합해 추출하기

# 1반의 수학,영어 성적 추출 
## 1반에 해당하는 행  + 수학/ 영어에 해당하는 변수 동시 추출 

exam.query("nclass == 1")[["math", "english"]] 
exam.query("nclass == 1") \
    [["math", "english"]] \ 
    .head()

exam.query("nclass == 1")[["math", "english"]].sort_values("math")
exam.query("nclass == 1")[2:4].sort_values("math")
 ##exam.query("nclass == 1")[2:4].sort_values("math",ascending = False) 
# 1반에 해당하는 행들 중 3,4번째 행들을 math 기준으로 오름차순 
# 인덱싱 행만 가능

--------------------------------------------------------------------------------

# 변수추가( chap4- 60번 참고)
exam = exam.assign(
    total = exam["math"] + exam["english"] + exam["science"],
    mean = (exam["math"] + exam["english"] + exam["science"])/3
    ) \
    .sort_values("total", ascending = False)

-----------------------------------------------------------------------------

# 정렬하기(chap4-100번대 참고)
exam.sort_values("math") # 오름차순 (작은 숫자부터)
exam.sort_values("math", ascending = False) # 내림차순 
exam.sort_values(["nclass", "english"], ascending = [True, False]) # 오름차순 안에 내림차순
    
# 정렬로 삼을 변수가 두개 이상이라면 []활용하여 리스트 만들어서 사용 
exam = exam.assign(
    total = exam["math"] + exam["english"] + exam["science"],
    mean = (exam["math"] + exam["english"] + exam["science"])/3
    ) \
    .sort_values(["total","mean"], ascending = False)    
    
exam.head()


# lambda 함수 사용하기
exam2 = pd.read_csv("data/exam.csv")

exam2 = exam2.assign(
    total = lambda x: x["math"] + x["english"] + x["science"],
    mean = lambda x: x["total"]/3
    ) \
    .sort_values("total", ascending = False)
exam2.head()

------------------------------------------------------------------------

# .groupby() + .agg() 콤보

# 집단별 요약 통계
# 그룹을 나눠 요약을 하는

## agg() 요약 통계량 함수 

exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass") \
     .agg(
         mean_math = ("math", "mean"),
         mean_eng = ("english", "mean"),
         mean_sci = ("science", "mean"),
     )



# 예제학습 153p/158p
import pandas as pd

df = pydataset.data("mpg")

# 1. 변수 이름 변경 했는지?
# 2. 행들을 필터링 했는지?
# 3. 새로운 변수를 생성했는지?
# 4. 그룹 변수 기준으로 요약을 했는지?
# 5. 정렬 했는지?

------------------------------------------------------

# 데이터 합치기 (167p)

test1 = pd.DataFrame({"id"     : [1, 2, 3, 4, 5], 
                      "midterm": [60, 80, 70, 90, 85]})

test2 = pd.DataFrame({"id"     : [1, 2, 3, 40, 5],
                      "final"  : [70, 83, 65, 95, 80]})

test1
test2

# 데이터를 가로로 쌓는 방법
# Left Join
total = pd.merge(test1, test2, how="left", on="id")
total
# Right Join
total = pd.merge(test1, test2, how="right", on="id")
total
# Inner Join 교집합
total = pd.merge(test1, test2, how="inner", on="id")
total
# Outer Join 합집합
total = pd.merge(test1, test2, how="outer", on="id")
total


# exam = pd.read_csv("data/exam.csv")
name = pd.DataFrame({"nclass": [1, 2, 3, 4, 5],
                     "teacher": ["kim", "lee", "park", "choi", "jung"]})

name
pd.merge(exam, name, how="left", on="nclass")


# 데이터를 세로로 쌓는 방법
score1 = pd.DataFrame({"id"     : [1, 2, 3, 4, 5], 
                      "score": [60, 80, 70, 90, 85]})

score2 = pd.DataFrame({"id"     : [6, 7, 8, 9, 10],
                      "score"  : [70, 83, 65, 95, 80]})
score1
score2
score_all=pd.concat([score1, score2])
score_all=pd.concat([score1, score2],axis =1)
score_all
 ## ch) lec5 : combined_vec = np.concatenate([str_vec, mix_vec]) 

test1
test2
-------------------------------------------------------------------------

예제 173p

mpg = pd.read_csv("data/mpg.csv")
mpg
mpg.loc[0:3] # 인덱싱은 3까지 , 넘파이 어레이일땐 미만 

fue1 = pd.DataFrame({"f1" : ["c", "d", "e","p", "r"],
                    "price_f1" : [2.35, 2.38, 2.11, 2.76, 2.22]
                    })





