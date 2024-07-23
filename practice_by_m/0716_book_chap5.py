import pandas as pd
import numpy as np 

#데이터 탐색
exam = pd.read_csv("data/exam.csv")
exam.head(10) # 함수 
exam.tail(10)
exam.shape  # 어트리뷰트
exam.info()
exam.describe()

#메서드 vs 어트리뷰트 (속성)
#메서드는 함수 , 어트리뷰트는 딸려있는 변수 

type(exam)
var=[1,2,3]
type(var)
exam.head()
exam
exam2= exam.copy()
exam2
exam2= exam2.rename(columns = {"nclass" : "class"})
exam2

exam2["total"] = exam2["math"] + exam2["english"] + exam2["science"]
exam2.head()

#200점 이상 pass 미만 fail 
exam2["test"] = np.where(exam2["total"]>=200,'pass','fail')
exam2.head()

# 빈도표 value_counts() 막대 그래프 plot.bar()

import matplotlib.pyplot as plt
count_test = exam2["test"].value_counts()
count_test.plot.bar(rot=0)
plt.show()
plt.clf()

# 200이상 A / 100 이상 B / 100미만 C
exam2["test2"] = np.where(exam2["total"]>=200,"A",
                 np.where(exam2["total"]>=100,"B","c"))
                       
exam2.head()
exam2["test2"].isin(["A","C"])


exam2["size"] = np.where(mpg[isin(["A","C"])
