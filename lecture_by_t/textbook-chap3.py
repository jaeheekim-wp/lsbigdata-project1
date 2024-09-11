# 교재 63 페이지

# seaborn 패키지 설치
# !pip install seaborn
# countplot() : Python의 데이터 시각화 라이브러리인 Seaborn에서 제공하는 함수, 
# 범주형 데이터의 개수를 세어 막대 그래프로 시각화하는 데 사용
#

import seaborn as sns
import matplotlib.pyplot as plt


var=["a", "a", "b", "c"]
var

sns.countplot(x=var)
plt.show()
plt.clf()

df = sns.load_dataset("titanic")
df

#빈도막대그래프 
#hue = 막대 색 다르게 표현하는 파라미터 
sns.countplot(data = df, x = "sex")
sns.countplot(data = df, x = "sex", hue="sex")
sns.countplot(data = df, x = "pclass", hue="pclass")
plt.show()

?sns.countplot
sns.countplot(data=df, x="class")
sns.countplot(data=df, x="class", hue="alive")
sns.countplot(data=df,
              x="survived",
              y="pclass")
plt.show()

# !pip install scikit-learn
import sklearn.metrics
# sklearn.metrics.accuracy_score()

from sklearn import metrics
# metrics.accuracy_score()

import sklearn.metrics as met
# met.accuracy_score()

score=[80,20,70]
sum_score= sum(score)
sum_score
#!/usr/bin/env python
import os
print(os.getcwd())

import pandas as pd

df_exam = pd.read_csv('exam.csv')
df_exam

# 혼자서 해보기

score = [80,90.70]
sum(score)
score_sum = sum(score)
score_sum
