*개인스터디

import pandas as pd
df= pd.DataFrame({'name': ['김지훈', '이유진', '박동현', '김민지'],
'english':[90,80,60,70],
'math':[50,60,100,20]}) 
df
df['english']
df['name']
sum(df['english'])
df['math']
sum(df['math'])
sum(df['english'])/4
sum(df['math'])/4

df= pd.DataFrame({'제품':['사과', '딸기', '수박'],
'가격':[1800,1500,3000],
'판매량':[24,38,13]})
# 평균값 
df['가격'].mean()
df['가격'].sum()
sum(df['가격'])/3
df['판매량']
sum(df['판매량'])/3

#엑셀 불러오기 

!pip install openpyxl
df_exam=pd.read_csv("C:/Users/USER/Documents/LS 빅데이터스쿨/lsbigdata-project1/exam.csv")
df_exam

sum(df_exam['english'])/20
sum(df_exam['english'])/len(df_exam)
sum(df_exam['science'])/20
sum(df_exam['science'])/len(df_exam)

df_exam_novar = pd.read_excel("C:/Users/USER\Documents/LS 빅데이터스쿨/lsbigdata-project1/excel_exam_novar.xlsx",
header = None)
df_exam_novar

df_midtern = pd.DataFrame({'english':[90,80,60,70],'math':[50,60,100,20],'nclass':[1,1,2,2]})
df_midtern

df_midtern.to_csv('output_newdata.csv')

#수업에서 진행 /63p

#패키지설치
#countplot() : Python의 데이터 시각화 라이브러리인 Seaborn에서 제공하는 함수, 
#범주형 데이터의 개수를 세어 막대 그래프로 시각화하는 데 사용
#!pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt  

#show 보여주기 / clf지우기 
plt.show()


var=['a','a','b','c']
var

sns.countplot(x=var, palette='viridis')
# 'viridis'는 Seaborn에서 제공하는 팔레트 중 하나
#'deep', 'muted', 'pastel', 'dark', 'colorblind'

#그래프 예시 1
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.plot(x, y)
plt.show()

#그래프 예시 1
import matplotlib.pyplot as plt

# 데이터 생성
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 그래프 생성
plt.plot(x, y, marker='o', linestyle='--', color='r')

# 제목과 라벨 추가
plt.title("Sample Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

#66p 타이타닉 데이터로 그래프 만들기 
df=sns.load_dataset('titanic')
df
sns.countplot(data=df, x='sex',hue='sex')
plt.show()

##파라미터 설정에 따른 기능 변화 
sns.countplot(data=df, x = 'class')
sns.countplot(data=df, x = 'class', hue ='alive')
sns.countplot(data=df, y = 'class', hue ='alive')
plt.show()
plt.clf()

#함수 사용법이 궁금할때 help함수
sns.countplot?

! pip install scikit-learn
import sklearn.metrics

sklearn.metrics.accuracy

from sklearn import metrics
metrics.accuracy_core()

from sklearn.metrics import accuracy_score
accuracy_score()

from sklearn import metrics as met
met.accuracy_score()

#패키지 함수 사용하기 
import pydataset
import pydataset as pyd
pyd.data()

df=pyd.data('mtcars')
df



