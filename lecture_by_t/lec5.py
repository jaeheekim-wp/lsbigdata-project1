import pandas as pd

import numpy as np

# 넘파이 벡터 슬라이싱 
# 벡터의 일부 추출시 대괄호[]사용
# 대괄호 안에는 추출하려는 요소의 '위치'나 '인덱스' 지정 

-------------------------------------------------------
# 벡터 슬라이싱 

# np.random.seed(2024) 
 ## 랜덤 숫자를 생성할 때, 특정 시드(seed)를 설정하면 매번 같은 순서의 숫자를 생성
 ## 랜덤값 고정 42 일반적, 숫자는 원하는 것으로
 ## 재현성을 위함

# np.random.randint 
 ##함수는 지정된 범위 내에서 정수를 생성.
 
np.random.seed(2024)
a = np.random.randint(1, 21, 10)
print(a)

# a = np.random.randint(1, 21, 10)
a = np.random.choice(np.arange(1, 4), 100, True, np.array([2/5, 2/5, 1/5]))
print(a)
sum(a == 1)
sum(a == 2)
sum(a == 3) 

#np.random.choice 함수는 주어진 배열에서 랜덤하게 값을 선택합니다.
#np.arange(1, 4): 선택할 값의 배열 [1, 2, 3].
#100: 선택할 값의 개수 (100개).
#True: 복원 추출 여부 (여기서는 복원 추출이므로 동일한 값이 여러 번 선택될 수 있습니다).
#np.array([2/5, 2/5, 1/5]): 각 값이 선택될 확률을 나타내는 배열.
-------------------------------------------------------------------------------

# 파이썬 인덱싱 특징

# 인덱스는 0부터 시작
# 양의 인덱스는 앞에서부터 세고, 음의 인덱스는 뒤에서부터 셉니다.
# 슬라이싱 구문 [start:stop:step]에서 stop은 포함되지 않습니다.

a = [0, 1, 2, 3, 4, 5]
a[:] #처음부터 끝
a[2:5] #2이상 5미만 
a[-2] #맨끝에서 두번째 
a[::2] #처음부터 끝까지, 스텝은 2
a[0:6:2] #0이상 6미만 스텝은 2  즉, 두번재 자리 부터 

start:stop:step

np.random.seed(42)
a = np.random.randint(1, 21, 10)
print(a)

# 두 번째 값 추출
print(a[1])

# 여러 값 동시 추출 시 (인덱싱 안에 리스트)
print(a[[0, 2, 4]])

# 인덱싱 중복 선택 가능
print(a[[1,1,3,2]])

# 비교 연산자 사용한 슬라이싱
b = a[a>7]
b

# 논리 연산자와 조건문 
# and &
c = a[(a>2) & (a<9)]
c
# or |
a =  np.array([1,2,3,4,16,17,18])
b = a[(a==4) | (a>15)]

# 연산자 추가 활용 
d = a[a!=8]
d

x= a[np.arange(1,11) % 2 == 1] 

# np.arange 활용하여 인덱스 벡터 생성 
# 이후 나머지 연산자 사용하여 2로 나눈 나머지가 1인 요소만 추출 -true값 추출
# 즉, 홀수번째 요소만 추출 가능 


# 1에서부터 1000사이 3의 배수의 합은?
sum(np.arange(1,1000)%3==0) #1
sum(np.arange(3,1001,3,)) #2
x= np.arange(1,1001) #3
sum(x[2:1000:3])

x= np.arange(0,1001) #4
sum(x[::3])


# 백터 제거하기 
a = [0, 1, 2, 3, 4, 5]
np.delete(a, 1)
np.delete(a, [1,3])

----------------------------------------------------

# 논리형 백터 
np.random.seed(42)
a = np.random.randint(1, 21, 10)
print(a)
a > 7
a[a > 7]

b = a[a > 3]
print(b)

b = a[(a > 2) & (a < 9)]
print(b)

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
# a[조건을 만족하는 논리형벡터]
a[(a > 2000) & (a < 5000)]

a>2000
a<5000

x = np.array([True, True, False])
y = np.array([True, False, False])
print(x & y) # 전부 참이여야 참 
print(x | y) # 하나라도 참이면 참 

#  필터링 통한 벡터 변경 
a = np.array([5,10,15,20,25,30])
a[a>=10] = 10
a
---------------------------------------------------
#!pip install pydataset
import pydataset

df=pydataset.data('mtcars')

np_df=np.array(df['mpg'])
type(np_df)

model_names = np.array(df.index)
type(model_names)

# 15 이상 25이하인 데이터 개수는?
sum((np_df >= 15) & (np_df <= 25))

# 15 이상 20이하인 자동차 모델은?
model_names[(np_df >= 15) & (np_df <= 20)]

# 평균 mpg 보다 높은(이상) 자동차 모델는?
model_names[np_df >= np.mean(np_df)]
model_names[np_df >= np_df.mean()]

# 평균 mpg 보다 낮은(미만) 자동차 모델는?
model_names[np_df < np.mean(np_df)]

# 15 작거나 22이상인 데이터 개수는?
sum((np_df < 15) | (np_df >= 22))


np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])
# a[조건을 만족하는 논리형벡터]
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)]
a[a > 3000] = 3000

# b[(a > 2000) & (a < 5000)]:
## 배열 a에서 값이 2000보다 크고 5000보다 작은 요소의 인덱스에 해당하는 b 배열의 요소를 선택
----------------------------------------------------------------------------------

# np.where()
# 조건을 만족하는 위치 탐색 
# 선택된 원소의 '위치'(인덱스)를 반환/ 즉 TRUE의 위치 반환 

a = np.array([1, 5, 7, 8, 10]) # 예시 배열
result = a < 7
result

result = np.where(a < 7)
type(result)

-------------------------------------------------------------------
#예제01

#처음으로 10000보다 큰 숫자가 나오는 위치는?
#숫자 위치와 그 숫자는 무었인가요
np.random.seed(2024)
a = np.random.randint(1, 26364, 1000)
type(a)
a[a>10000][0]
# np.where 활용
a[np.where(a>10000)][0]

# a[a > 10000][0]
 ##01.a > 10000은 a 배열에서 10000보다 큰 값이 있는 위치를 True로 표시하는 논리배열.
 ##02.a[a > 10000]은 a 배열에서 10000보다 큰 값들만 추출한 새로운 배열을 만듭니다.
 ##03.[0]은 이 새로운 배열에서 첫 번째 값을 선택.

# np.where(a > 10000)
 ##01.np.where(a > 10000)은 a 배열에서 10000보다 큰 값들의 '인덱스' 반환.
 ##02.a[np.where(a > 10000)]은 이 인덱스를 사용하여 a 배열에서 10000보다 큰 값들을 추출한 새로운 배열을 만듭니다.
 ##03.[0]은 이 새로운 배열에서 첫 번째 값을 선택합니다.
 
--------------------------------------------------------------- 
#예제02

# 처음으로 22000보다 큰 숫자 나왔을때,
# 숫자 위치와 그 숫자는 무엇인가요?

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
x = np.where(a > 22000)
type(x)
my_index = x[0][0]
my_index
a[my_index]
a[10]

#np.where(a > 22000)는 a 배열에서 22000보다 큰 값들의 '인덱스'를 반환.
 ## 결과: x는 조건을 만족하는 요소들의 인덱스 배열을 포함하는 '튜플'.
 
#x[0]는 조건을 만족하는 요소들의 인덱스 배열.
#x[0][0]는 조건을 만족하는 첫 번째 요소의 인덱스 추출.

#my_index는 a 배열에서 22000보다 큰 첫 번째 요소의 인덱스.
#a[my_index]는 a 배열에서 my_index에 해당하는 요소 추출.
 ## 결과: a 배열에서 22000보다 큰 첫 번째 값을 반환.

--------------------------------------------------------------
#예제03

# 처음으로 10000보다 큰 숫자들 중
# 50번째로 나오는 숫자 위치와 그 숫자는 무엇인가요?

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
x=np.where(a > 10000)
type(x)
a[x[0]]  # 튜플 안에 있는 어레이 꺼내기 
type(a)
a[x[0][49]]
 # 궁금증 : 21052의 위치 인덱스를 뽑고 싶으면?

#np.where(a > 10000)는 a 배열에서 10000보다 큰 값들의 '인덱스'를 반환.
 ## 결과: x는 조건을 만족하는 요소들의 인덱스 배열을 포함하는 '튜플'.

#x[0]는 a > 10000 조건을 만족하는 요소들의 인덱스 배열.
#a[x[0]]는 a 배열에서 이 인덱스들에 해당하는 값들을 선택.
 ## 결과: a 배열에서 10000보다 큰 값들의 배열을 반환합니다.

#x[0][49]는 조건을 만족하는 인덱스 배열에서 50번째 인덱스 추출.
#a[x[0][49]]는 a 배열에서 이 50번째 인덱스에 해당하는 값을 가져옵니다.
 ## 결과: 10000보다 큰 50번째 값을 반환.

------------------------------------------------------------------
#예제04

# 50보다 작은 숫자들 중
# 가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가요?

np.random.seed(2024)
a = np.random.randint(1, 50, 10)
a
x=np.where(a < 30)
a[x[0][-1]]
a[9]

# 조별 과제 중 다른 팀 where활용안 - 막대 컬러 지정에 사용 
# bar_colors = np.where(oecd["country"]=="Korea","red","dodgerblue")
# sns.barplot(data=oecd.sort_values("real_wage",ascending=True), \
# x='country', y='real_wage', palette=bar_colors)

---------------------------------------------------------------------------
# 벡터 함수 
 #사용하기 예제

a = np.array([1, 2, 3, 4, 5])
sum_a = np.sum(a) # 합계 계산
mean_a = np.mean(a) # 평균 계산
median_a = np.median(a) # 중앙값 계산
std_a = np.std(a, ddof=1) # 표준편차 계산
sum_a, mean_a, median_a, std_a

---------------------------------------------
# nan

# 빈 칸 나타내는 법
# 데이터가 정의되지 않은 np.nan / not a number
# 실제 값을 가지고 있지 않지만 벡터의 길이나 타입 유지를 위해 존재 

a = np.array([20, np.nan, 13, 24, 309])

# 타입: float. numpy 라이브러리에서 제공하는 상수
# 사용처: 주로 데이터 분석에서 결측값(missing value)을 나타내기 위해 사용
# 비교: np.nan과의 비교는 직접적으로 == 연산자를 사용할 수 없고, 
# numpy의 np.isnan() 함수를 사용해야 합니다.

# 벡터 안에 nan이 있는 경우, 계산값은 nan으로 출력
np.mean(a)

# 방지 차원의 nan무시옵션
np.nanmean(a)

-------------------------------------------------

# 빈칸 제거 방법 

# isnan()
# 벡터 a 의 원소가 nan인지를 알려주는 함수
# nan 인 경우 - true / 아닌 경우 false

np.isnan(a)

~np.isnan(a)
# ~ : 반전,0은 1로 1은 0으로 
# 즉, True였던 nan값을 False로 바꿔준다. 

a_filtered = a[~np.isnan(a)] # True인 값 (nan 제외) 인 것만 가져온다다
a_filtered

# 수치 연산 가능 
a + 3
np.nan + 3
np.mean(a)
np.nanmean(a)

# np.nan_to_num 함수로 NaN 값을 0으로 대체
b = np.nan_to_num(a, nan=0)

print("Original array:", a)
print("Array after replacing NaN with 0:", b)
----------------------------------------------------

# None과의 차이 

# None은 값이 없음을 나타내는 특수 상수 
# 타입: Nonetype
# 수치 연산 불가
a = None
a + 1
b = np.nan
b + 1
----------------------------------------------------

# 벡터 합치기 

# 벡터는 숫자 뿐만 아니라 같은 타입의 정보 (숫자,문자) 를 묶어놓은 것 
# 즉 숫자면 숫자, 문자면 문자이기만 하면 묶을 수 있다.

str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]


# 섞인 타입일 경우 자동으로 통일할 수 잇는 타입(문자) 정보로 바꿔서 저장 
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

combined_vec = np.concatenate([str_vec, mix_vec])
combined_vec

-------------------------------------------------------------------------

# 함수 붙이기 

# 함수들 세로로 붙이기
col_stacked = np.column_stack((np.arange(1, 5), 
                               np.arange(12, 16)))
col_stacked

# 함수들 가로로 붙이기 
row_stacked = np.vstack((np.arange(1, 5), 
                         np.arange(12, 16)))
row_stacked


# 길이 맞춰서 세로로 붙이기 
uneven_stacked = np.column_stack((np.arange(1, 5), 
                                  np.arange(12, 18)))
uneven_stacked


vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)

np.resize(vec1, len(vec2))
vec1 = np.resize(vec1, len(vec2))
vec1


# 길이 다른 벡터 붙이는 방법 

uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked


vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.vstack((vec1, vec2))
uneven_stacked

---------------------------------------------

# 예제

# 홀수번째 원소 추출
a = np.array([12, 21, 35, 48, 5])
a[0::2] # 0부터 끝까지 2간격 
a[1::2] # 1부터 끝까지 2간격 
a[a % 2 == 1] a를 2로 나눈 나머지가 1인 값들 추출 

# 최대값 찾기
a = np.array([1, 22, 93, 64, 54])
a.max()

# 중복값 제거 
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
a
np.unique(a)


# 원소 번갈아가면서 합치기
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
a
b
# np.array([21, 24, 31, 44, 58, 67])


## 01. 그릇 먼저 만들기 
x = np.empty(6)
x

## 02. 홀수
# x[[0, 2, 4]] = a
x[0::2] = a
x

## 03. 짝수
# x[[1, 3, 5]] = b
x[1::2] = b
x

