a=(1,2,3) # a=1,2,3
a

a=[1,2,3]
a

# soft copy
b = a
b

a[1]=4
a
b
# 
# id : 객체의 고유 식별자 반환, 즉 객체의 주소를 알려줌 
id(a)
id(b)

# deep copy
# a와 b가 같은 주소를 할당받았지만 같은 변경을 원하지 않을 때 
a=[1,2,3]
a

id(a)

b=a[:]
b=a.copy()

id(b)

a[1]=4
a
b

# 수학함수
# 온라인PDF 03의 1P 참고 
import math 
x=4
math.sqrt(x)

exp_val = math.exp(5)
exp_val

(2 * math.sqrt(2 * math.pi))**-1

def my_normal_pdf(x, mu, sigma):
  part_1=(sigma * math.sqrt(2 * math.pi))**-1
  part_2=math.exp((-(x-mu)**2) / (2*sigma**2))
  return part_1 * part_2

my_normal_pdf(3, 3, 1)


def normal_pdf(x, mu, sigma):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    factor = 1 / (sigma * sqrt_two_pi)
    return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def my_f(x, y, z):
    return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)

my_f(2, 9, math.pi/2)


def my_g(x):
    return math.cos(x) + math.sin(x) * math.exp(x)

my_g(math.pi)
--------------------------------------------------------
# Ctrl + Shift + c : 커멘트 처리
# !pip install numpy
import pandas as pd
import numpy as np

# PANDAS 데이터 프레임을 위한 파이썬 라이브러리 
# NUMPY 과학 계산,수치 데이터 을 위해 사용되는 파이썬 라이브러리 

# 넘파이 
# 고성능의 다차원 배열 (ndarry) 지원, 벡터화된 연산 수행
# 데이터 조작,정제,필터링,변형 등 

# 백터란:간단히 말해 숫자의'리스트'
# 소괄호() 안에 대괄호 [] 
# 이 리스트는 크기와 방향을 가지고 있음(수학/물리학에서 주로 사용)
# 프로그래밍에서 벡터는 주로 데이터를 효율적으로 저장하고 조작하는 데 사용

--------------------------------------------------------------------
# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

# 넘피어레이 타입 확인
type(a)

a[3]
a[2:]
a[1:4]

a=[1, 3, 4]

# 빈 배열 선언 후 채우기 empty/zeros 동일 
b = np.empty(3)
b = np.zeros(3)
b
b[0]=1
b[1]=4
b[2]=10

b

#np.array() 함수를 직접 사용
#np.arange() 일정한 '간격'의 숫자 배열 생성
#np.linspace() 지정된 범위를 균일하게 나눈 숫자 배열 생성_'갯수'
#np.repeat() 함수, 값을 반복해서 벡터 만들기

# np.array :  함수 직접 지정 

vec1 = np.array([1, 2, 3, 4, 5])


# np.arange : 함수 사용하여 일정 간격의 숫자 배열 생성 

vec1 = np.arange(100)
vec1 = np.arange(1, 100, 0.5) 
vec1
# 0.5 간격으로 
# 1 이상(포함) 100 미만 (미포함)

# -100 부터 0까지
vec2=np.arange(0, -100, -1)
vec3=np.arange(0, 100)
vec2


# np.linspace : 함수 사용하여 지정 갯수의 숫자 배열 형성, 균일 간격
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# 갯수 미지정시 기본값은 50 
l_space1 = np.linspace(1, 100, 100)
l_space1 = np.linspace(1, 100)



# endpoint : t=stop 값 포함 f-stop값 제외 
linear_space2 = np.linspace(0, 1, 5, 
                            endpoint=False)
linear_space2

linear_space3 = np.linspace(1, 100, 20, endpoint=True, retstep=True, dtype=None)
linear_space3
#마지막 100 포함, 결과와 함께 샘플 간격 반환

## tip. 앞에 물음표 붙이면 해당 함수에 대한 설명 
?np.linspace


#배열 반복 (반복할 백터 , 반복 횟수수)
repeat_test = np.repeat([1,2,3],2)
repeat_test

# repeat vs. tile
vec1=np.arange(5)
np.repeat(vec1, 3)
np.tile(vec1, 3)

vec1=np.array([1, 2, 3, 4])
vec1 + vec1

max(vec1)
min(vec1)
sum(vec1)
vec1.min()
vec1.max()
vec1.sum()


# 35672 이하 홀수들의 합은?
sum(np.arange(1, 35673, 2))
x=np.arange(1, 35673, 2)
x.sum()



#넘파이 백터 길이 재는 법 
len(x)
x.shape
x.size

length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수


# len()함수 활용하기 
# 배열의 첫번째 차원의 길이 반환 
# 1차원 배열의 경우, 배열의 요소 수 의미 
a = np.array([1,2,3,4,5])
len(a)

--------------------------------------------
# shape속성 활용하기 ( 함수아님)
# 배열의 각 차원의 크기를 튜플 형태로 반환
# 이를 통해 배열의 전체 크기 확인 
a = np.array([1,2,3,4,5])
a.shape

---------------------------------------------
# size속성 활용하기( 함수아님)
# 배열의 모든 요소 갯수 반환 
a = np.array([1,2,3,4,5])
a.size

-----------------------------------------------
# 다차원 배열의 길이 재기
b = np.array([[1, 2, 3], [4, 5, 6]])
len(b)
b.shape
b.size

------------------------------------------------
# 넘파이 활용하여 벡터연산하기
a = np.array([1,2,3])
b = np.array([4,5,6])
a + b
a - b
a * b
a / b
a % b

a.shape
b.shape

-------------------------------------------------------
# 넘파이 브로드캐스팅 
# 길이가 다른 배열 간의 연산을 가능하게 해주는 매커니즘
# 작은 배열이 큰 배열의 길이에 맞추어 자동으로 확장 연산 
# 연산 가능 경우
## 01.차원의 크기가 같을 때 
## 02.차원 중 하나의 크기가 1인 경우 
-------------------------------------------------------
#1차원 브로드 캐스팅

## 01. 차원의 크기 맞춰서 계산 
a=np.array([1,2])
b=np.array([1,2,3,4])
a + b  

np.tile(a, 2) + b
np.repeat(a, 2) + b


## 02. 차원 중 하나의 크기가 1인 경우  
a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b


# 논리형 값 활용하기 (참고 : "==" 동등하다 / "=" 변수에 할당하다 )
# 논리형 값으로 반환 시 갯수 확인에 용이 
a == 1
b == 3

# 35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
sum((np.arange(1, 35672) % 7) == 3)

# 10 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
sum((np.arange(1,10) % 7) == 3)

---------------------------------------

# 2차원 브로드 캐스팅 

#01 가로 벡터 더하기 

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
 
matrix.shape 
# (행, 열)

# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector.shape

# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)


# 02 세로 벡터로 더하기 

# 2차원 배열 생성 
matrix = np.array([[ 0.0, 0.0, 0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
matrix.shape

# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector.shape

result = matrix + vector
print("브로드캐스팅 결과:\n", result)



a = 3,
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4,1)
vector
vector.shape
result = a + vector
result


x=np.array([1.0, 2.0, 3.0, 4.0])
y=np.array([[[1.0, 2.0, 3.0, 4.0]]])
x.shape
y.shape 
# 질문하기 
