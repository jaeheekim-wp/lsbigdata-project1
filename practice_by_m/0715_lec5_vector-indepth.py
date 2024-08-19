import numpy as np 

# 벡터 슬라이싱 예제, a를 랜덤하게 채움

np.random.seed(42)

# 1이상 21미만의 10개 숫자 랜덤 추출 
np.random.seed(2024) #랜덤값 고정 42 일반적, 숫자는 원하는 것으로 
a = np.random.randint(1, 21, 10)
print(a)


a = np.random.choice(np.arange(1,21), 10, False)
print(a)
a = np.random.choice(np.arange(1,4), 100, True, np.array([2/5,2/5,1/5]))
print(a)
sum(a==1)
sum(a==2)
sum(a==3)

print(a)
# 두 번째 값 추출
print(a[1])

a = [0, 1, 2, 3, 4, 5]
a[:] #처음부터 끝
a[2:5] #2이상 5미만 
a[-2] #맨끝에서 두번째 
a[::2] #처음부터 끝까지, 스텝은 2
a[0:6:2] #0이상 6미만 스텝은 2  즉, 두번재 자리 부터 

start:stop:step

#1에서부터 1000사이 3의 배수의 합은?
sum(np.arange(1,1000)%3==0) #1
sum(np.arange(3,1001,3,)) #2
x= np.arange(1,1001) #3
sum(x[2:1000:3])

x= np.arange(0,1001) #4
sum(x[::3])


print(a[[0, 2, 4]])
np.delete(a, 1)
np.delete(a, [1,3])

#논리형 백터 

np.random.seed(42)
a = np.random.randint(1, 21, 10)
print(a)
a > 3
a[a > 3]

b = a[a > 3]
print(b)

b = a[(a > 2) & (a < 9)]
print(b)

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
a
a[(a>2000)&(a <5000)]

a>2000
a<5000

x = np.array([True, True, False])
y = np.array([True, False, False])
print(x & y)

!pip install pydataset
import pydataset

df=pydataset.data('mtcars')
df
np_df=np.array(df['mpg'])
np_df

model_names=np.array(df.index)
model_names

#15이상 25이하인 데이터 갯수는
sum((np_df>= 15) & (np_df<=25))

#평균 
sum(np_df>= np.mean(np.df))
#15작거나 22이상인 데이터 갯수 

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b= np.array(['A','B','C','F','W'])
a[(a>2000)&(a <5000)]
b[(a>2000)&(a <5000)]
a
a[a>3000] =3000 
# 값 변경 방법 300 이상이면 3000으로 작성 
a

#조건문 혼합하기
import numpy as np
a = np.array([1, 2, 3, 4, 16, 17, 18]) # 예시 배열
result = a[(a == 4) & (a > 15)]
print(result)
#값이 4인 원소와 15보다 큰 원소 두 가지 조건을 모두 만족하
#는 원소는 없으므로, 빈 배열을 반환



#필터링 통한 벡터 변경 
import numpy as np
a = np.array([5, 10, 15, 20, 25, 30]) # 예시 배열
a[a >= 10] = 10
a

#np.where()선택된 원소의 위치를 반환/ 즉 TRUE의 위치 반환 
import numpy as np
a = np.array([1, 5, 7, 8, 10]) # 예시 배열
result = a < 7
result

result = np.where(a < 7)
result

np.random.seed(2024)
a = np.random.randint(1, 100, 10)
a
a<50
np.where(a<50)

np.random.seed(2024)
a = np.random.randint(1, 26364, 1000)
a
#처음으로 10000보다 큰 숫자가 나오는 위치는?
#숫자 위치와 그 숫자는 무었인가요
a[a>10000][0]
a[np.where(a>5000)]

a[np.where(a>10000)][0]

#처음으로 22000보다 큰 숫자가 나오는 위치는?
#숫자 위치와 그 숫자는 무엇인가요

np.random.seed(2024)
a = np.random.randint(1, 26364, 1000)
a
x=np.where(a>22000)
x
type(x)
my_index= x[0][0]
#넘파이 어레이(하나의 백터 ) 로 들어가서 그 안에 0번째 원소를 꺼내는 것 
a[my_index]
a[np.where(a>22000)][0]


#처음으으로 10000보다 큰 숫자가 나오는 위치는?
#50번째로 나오는 숫자와 그 위치는 
x=np.where(a>10000)
x
type(x)
my_index= x[0][0]
a[my_index]


#500보다 작은 숫자들중
#가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가요

np.random.seed(2024)
a = np.random.randint(1, 26364, 1000)
a
x=np.where(a<500)
x
a[x[0][-1]]

# 벡터 함수 사용하기 예제
a = np.array([1, 2, 3, 4, 5])
sum_a = np.sum(a) # 합계 계산
mean_a = np.mean(a) # 평균 계산
median_a = np.median(a) # 중앙값 계산
std_a = np.std(a, ddof=1) # 표준편차 계산
sum_a, mean_a, median_a, std_a

# np.nan는 정의 되지 않은 값(not a number)을 나타냅니다
import numpy as np
a = np.array([20, np.nan, 13, 24, 309])
a
a + 3
np.nan +3
np.mean(a)
np.nanmean(a) # nan 무시 
np.nan_to_num(a, nan=0)

#타입: float 타입입니다. numpy 라이브러리에서 제공하는 상수
#사용처: 주로 데이터 분석에서 결측값(missing value)을 나타내기 위해 사용
#비교: np.nan과의 비교는 직접적으로 == 연산자를 사용할 수 없고, 
#numpy의 np.isnan() 함수를 사용해야 합니다.

#np.isnan(a) 
#nan 부분을 True로 
#벡터 a의 원소가 nan인지를아닌지를 알려주는 함수입니다. 
#nan인 경우 True를 반환하고, 그렇지 않은 경우 False를 반환

#None 
#변수 초기화, 함수 반환 값, 조건문, 기본 인자값 등에 사용.
a=None
b=np.nan
b+1
a+1


~ 반대 :
~np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered

#벡터 합치기
str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

#자동으로 통일할 수 있는 타입(문자) 정보로 바꿔서, 
#벡터로 저장하는 것
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

#여러 개의 벡터들을 하나로 묶을 수 있는 방법
combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec = np.concatenate([str_vec, mix_vec])
combined_vec 
#튜플이던 리스트이던 결과는 동일 


#np.column_stack() 함수는 벡터들을 세로로
col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

#np.row_stack() 함수는 벡터들을 가로로
row_stacked = np.row_stack((np.arange(1, 5), np.arange(12, 16)))
row_stacked
#vstack으로 권장 
row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked
type(row_stacked)


#길이가 다른 벡터 합치기
uneven_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 18)))
uneven_stacked

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1
#numpy.resize(arr, new_shape) 함수는 주어진 배열 arr을 
#new_shape으로 크기를 조정하여 새로운 배열을 반환하며, 
#원래 배열의 요소가 부족하면 반복해서 채우고, 넘치면 잘라냅니다

uneven_stacked = np.column_stack((vec1,vec2))
uneven_stacked

#연습문제 
#01홀수번째 원소 
a = np.array([12, 21, 35, 48, 5])
a[0:2]
a[1:2]

#02최대값 
a = np.array([1, 22, 93, 64, 54])
a.max()

#03중복값삭제
#np.unique()는 데이터 분석에서 데이터의 고유한 값들을 찾거나, 
#중복된 데이터를 제거할 때 매우 유용한 함수입니다.
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

#04번갈아합쳐서 새로운 백터만들기 
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])

x=np.empty(6) #그릇 만들기기
x

#홀수
x[[0,2,4]]= a
x[0::2] = a
x

#짝수
x[[1,3,5]] =b
x[1::2]=a
x

#05.
import numpy as np

# Step 1: 벡터 a와 b 정의
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])

# Step 2: a 벡터의 마지막 값을 제외
a_trimmed = a[:-1]  # a[:-1]는 [1, 2, 3, 4]를 반환

# Step 3: 두 벡터를 더하기
result = a_trimmed + b  # [1, 2, 3, 4] + [6, 7, 8, 9]는 [7, 9, 11, 13]을 반환

# 결과 출력
print(result)  # 출력: [ 7  9 11 13]


a = np.array([1, 2, 3, 4, 5])
a + 5

