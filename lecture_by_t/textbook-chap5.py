import pandas as pd
import numpy as np

# 데이터 탐색 함수

# 메서드 - 만들어진 변수에 사용되는 함수/ 변수에 사용되는 '기술'
# 어트리뷰트 - 변수가 지니고 있는 값 / 변수가 가진 '능력치'(112p)

# head()
# tail()
# shape - 어트리뷰트 
# info()
# describe()

exam = pd.read_csv("data/exam.csv")
exam.head(10)
exam.tail(10)
exam.shape # (행,열) 
exam.info()
exam.describe()
 # 문자로 된 변수의 요약 통계량을 함꼐 출력하려면 include = all
exam

type(exam)
var=[1,2,3]
type(var)
exam.head() #데이터프레임에서만 가능 
# var.head() 에러

# 파일 복제하기 
exam2 = exam.copy()

# 변수명 변경
exam2 = exam2.rename(columns={"nclass" : "class"})
exam2

# 파생변수 추가하기
exam2["total"] = exam2["math"] + exam2["english"] + exam2["science"]

 ## total 변수의 평균 구하기
sum(exam2["total"]) / len(exam2)
exam2["total"].mean()

exam2.head()

# 조건문 활용해 변수 만들기 
exam2["test"] = np.where(exam2["total"] >= 200, "pass", "fail")
# 200 이상: pass
# 200 미만: fail

 ## 중첩 조건문 활용 
exam2["test2"] = np.where(exam2["total"] >= 200, "A",
                 np.where(exam2["total"] >= 100, "B", "C"))
exam2
exam2.head()

# 200 이상: A
# 100 이상: B
# 100 미만: C

exam2["test2"].isin(["A", "C"]) 

---------------------------------------------------------

mpg = pd.read_csv("data/mpg.csv")
mpg
mpg["total"] = (mpg["cty"] + mpg["hwy"]) / 2
mpg["test"] = np.where(mpg["total"]>=20, "pass", "fail")
mpg["grade"] = np.where(mpg["total"]>=30, "A",
               np.where(mpg["total"]>=20, "B","C"))
               
# 목록에 해당하는 행으로 변수만들기(128p)
mpg["size"] = np.where(mpg["category"].isin(["compact","subcompact","2seater"]), "small", "large")
  
  ---------------               
# 그래프 만들기 
import matplotlib.pyplot as plt


# plot.hist() 히스토그램 만들기 
# 히스토그램- 값의 빈도를 막대길이로 표현
# 분포도 확인 
mpg["total"].plot.hist()
plt.show()


# value_counts() 빈도표 
 ## 기본: 내림차순 (많은 순서대로)
count_test = mpg["test"].value_counts()
count_test

count_grade = mpg["grade"].value_counts() 
count_grade

count_size = mpg["size"].value_counts() 
count_size

 ## sort_index() 알파벳 순으로 정렬 - 메서드 체이닝 활용(126p)
count_grade = mpg["grade"].value_counts().sort_index()
count_grade

# plot.bar() 빈도 막대그래프 
count_test.plot.bar(rot=0)
plt.title('test result')
plt.xlabel('test')
plt.ylabel('Value')
plt.show()
plt.clf()

count_grade.plot.bar(rot=0)
plt.title('grade result')
plt.xlabel('grade')
plt.ylabel('Value')
plt.show()
plt.clf()

