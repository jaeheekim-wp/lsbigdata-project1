# 튜플: 괄호 없이도 동일 
# 왜 이렇게 사용 가능한지 : 
#빠른실행..?
a=(1,2,3)
a

# soft copy 
a=[1,2,3]
a
b=a
b
a[1]=4
a
b
id(a)
id(b)

#b=a라는 식을 작성하게되면, 일반적으로 b가a와 동일한 내용을 갖기를 바라지만,
#대체로 a가 변경될때 b도 같이 변경되기를 바라는 경우는 아니다.
#같은 리스트를 참조할 뿐, 동일 및 같다는 뜻 아님! 
#즉, 원하지 않는 변경이 생길때는 deep copy 활용 

#deep copy
a=[1,2,3]
a

id(a)

b=a[:]
b=a.copy()
id(b)

a[1]=4
a
b

a=[1,2,3]
b= copy.deepcopy(a)
a[0]=4
print(a)
print(b)

#수학함수활용하기 
import math

x=4
math.sqrt(x)

exp_val = math.exp(5)
print("e^5의 값은:", exp_val)

log_val = math.log(10, 10)
print("10의 밑 10 로그 값은:", log_val)

fact_val = math.factorial(5)
print("5의 팩토리얼은:", fact_val)

#예제: 𝜇 = 0, 𝜎 = 1에서 𝑥 = 1의 확률밀도함수 값 계산
def normal_pdf(x, mu, sigma):
 sqrt_two_pi = math.sqrt(2 * math.pi)
 factor = 1 / (sigma * sqrt_two_pi)
 return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def my_normal_pdf(x,mu,sigma):
  part_1=(sigma * math.sqrt(2*math.pi))**-1
  part_2=math.exp((-(x-mu)**2)/ (2*sigma**2))
  return 

 my_normal_pdf(3,3,1)
   

def my_f(x,y,z) :
   return(x ** 2 + math.sqrt(y) + math.sin(z)) * math.exp(x)
my_f(2,9, math.pi/2)

x = math.pi
g_value = math.cos(x) + math.sin(x) * math.exp(x)
print("삼각함수와 지수를 결합한 수식 값은:", g_value)    

#def +  tab 
def fname(`indent('.') ? 'self' : ''`):
    """docstring for fname"""
    # TODO: write code...
    
    
#fcn  snippet 등록
def fname(input):
    contents
    return
    contents
    return
#snippet 단축키 ; 쉬프트+ 스페이스 
#fcn치고 단축키 
def fname(input):
    contents
    return 

# pandas/numpy - snippet 등록 
import pandas as pd
import numpy as np 
# pd + shift +space
# np +shift _space 

#numpy 활용
#ctrl +shift +c : 커맨드 처리 
!pip install numpy
import numpy as np 

#Python에서 벡터와 같은 '수치 데이터'를 다루는 데 매우 효율적인 라이브러리
#백터란:간단히 말해 숫자의 리스트
#이 리스트는 크기와 방향을 가지고 있음(수학/물리학에서 주로 사용)
#프로그래밍에서 벡터는 주로 데이터를 효율적으로 저장하고 조작하는 데 사용됩니다.

#숫자형
a = np.array([1, 2, 3, 4, 5])
#문자형
b = np.array(["apple", "banana", "orange"]) 
#논리형 
c = np.array([True, False, True, True])
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)

#벡터를 생성하는 방법
#1. 빈 배열 선언 후 채우기 / np.empty() 또는 np.zeros() 함수
#2. 배열을 생성하면서 채우기

# 빈 배열 생성
x = np.empty(3) # 0부터 
print("빈 벡터 생성하기:", x)
# 배열 채우기
x[0] = 3
x[1] = 5
x[2] = 3
print("채워진 벡터:", x)

#한번 넘파이로 정의하면 넘파이로 계속 
type(a)
a[3]
a[2:]
a[1:4]

b = np.empty(3)
b
b[0]=1
b[1]=4
b[2]=10
b
b[2]

#np.array() 함수를 직접 사용
#np.arange() 일정한 '간격'의 숫자 배열 생성
#np.linspace() 지정된 범위를 균일하게 나눈 숫자 배열 생성_'갯수'
#np.repeat() 함수, 값을 반복해서 벡터 만들기

vec1=np.array([1,2,3,4,5])
vec1=np.arange(100)
vec1=np.arange(1,100)
vec1=np.arange(1,101,0.5) 
#1이상 101미만/ 0.5간격으로 
vec1


linear_space1 = np.linspace(0, 1, 5)
print("0부터 1까지 5개 원소:", linear_space1)
#균등 간격 

linear_space2 = np.linspace(0, 1, 5, endpoint=False)
print("0부터 1까지 5개 원소, endpoint 제외:", linear_space2)
#endpoint 옵션 변경
#0부터 1까지 총 5개의 요소로 구성되지만, 1은 포함하지 않는 배열을 생성

#-100부터 0까지
vec2=np.arange(-100,1)
vec2

vec2=np.arange(0,-100)
vec2 
#왼쪽(마이너스)으로는 이동 불가 

vec2=np.arange(0,-100,-1)
vec2
#로는 가능 

vec3=np.linspace(0,-100,5)
vec3

#repeat: 개별 원소를 원하는 수만큼 반복 
vec1=np.arange(5)
np.repeat(vec1,5)
vec1

#1 단일값 반복 (숫자 8을 4번 반복)
repeated_vals = np.repeat(8, 4)
print("Repeated 8 four times:", repeated_vals)

#2 배열 반복 (배열 [1, 2, 4]를 2번 반복)
repeated_array = np.repeat([1, 2, 4], 2)
print("Repeated array [1, 2, 4] two times:", repeated_array)

#3 각 요소 반복(배열 [1, 2, 4]의 각 요소를 각각 1, 2, 3번 반복)
repeated_each = np.repeat([1, 2, 4], repeats=[1, 2, 3])
print("Repeated each element in [1, 2, 4] two times:", repeated_each)

#4 벡터 전체를 반복
#tile: 전체 백터를 뭉텅이로 반복 
np.tile(vec1,3)
repeated_whole = np.tile([1, 2, 4], 2)
print("벡터 전체를 두번 반복:", repeated_whole)

vec1=np.array([1,2,3,4])
vec1*2
vec1/3
vec1+vec1
vec1+vec1

min(vec1)
max(vec1)
sum(vec1)

#Q.35672이하 홀수들의 합은 ?
np.arange(1,35673,2)

# 같은 값 다른 코드 
sum(np.arange(1,35673,2))
np.arange(1,35673,2).sum()
x=np.arange(1,35673,2)
x.sum()

#넘파이 벡터 길이 재는 방법

#1. len 첫번째 차원의 길이
#2. shape 각 차원의 크기 > 튜플 
#3. size전체 갯수 

len(x)
x.shape

b=np.array([[1,2,3],[4,5,6]])
len(b)
b.shape
b.size

#len() 함수는 배열의 첫 번째 차원의 길이를 반환합니다
#용도: 리스트, 문자열, 튜플, 사전 등 여러 종류의 자료형의 길이(첫 번째 차원)를 반환
#반환값: 자료형의 첫 번째 차원의 크기를 정수로 반환합니다

# 1차원 리스트 배열
a = np.array([1, 2, 3, 4, 5])
len(a)

#문자열 배열 
my_string = "Hello"
print(len(my_string))

#2차원 리스트 배열 
my_2d_list = [[1, 2, 3], [4, 5, 6]]
print(len(my_2d_list))  # 출력: 2

#shape 속성은 배열의 각 차원의 크기를 튜플 형태로 반환합니다.
#용도: numpy 배열의 각 차원에서의 크기를 튜플로 반환합니다.
#반환값: 배열의 각 차원의 크기를 나타내는 튜플을 반환합니다.

# 1차원 배열
a = np.array([1, 2, 3, 4, 5])
a.shape

# 2차원 배열
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape) 

#size 배열의 총 요소 수를 반환하는 데 사용됩니다. 
#이를 통해 배열에 몇 개의 요소가 있는지 쉽게 알 수 있습니다
#용도: numpy 배열의 전체 요소 수를 반환합니다.
#반환값: 배열에 포함된 모든 요소의 개수를 정수로 반환합니다.

# 1차원 배열
a = np.array([1, 2, 3, 4, 5])
a.size

#2차원 배열
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.size)  # 출력: 6

#3차원 배열
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(c.size)  # 출력: 8

b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수
length, shape, size

#NumPy를 사용하여 벡터 연산하기
import numpy as np

# 벡터 생성
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# 벡터 간 덧셈
add_result = a + b
print("벡터 덧셈:", add_result)
# 벡터 간 뺄셈
sub_result = a - b
print("벡터 뺄셈:", sub_result)
# 벡터 간 곱셈
mul_result = a * b
print("벡터 곱셈:", mul_result)
# 벡터 간 나눗셈
div_result = a / b
print("벡터 나눗셈:", div_result)
# 벡터 간 나머지 연산
mod_result = a % b
print("벡터 나머지 연산:", mod_result)

a=np.array([1,2])
b=np.array([1,2,3,4])
a+b
#서로 길이가 맞지 않으면 더할 수 없다

np.tile(a,2)+b
np.repeat(a,2)+b
x = np.array([1, 2, 4, 5])
y = x * 2
print("상수 곱셈:", y)

#3과 동일한 값이면 true로 전환 
b==3 

#10보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
np.arange(1,10)
(np.arange(1,10) % 7)==3
sum((np.arange(1,10) % 7)==3) 
#true를 1로 인식

#35672보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?

(np.arange(1,35672) % 7)==3
sum((np.arange(1,35672) % 7)==3)

#브로드캐스팅(Broadcasting) 개념
#길이가 다른 배열 간의 연산을 가능하게 해주는 강력한 메커니즘

a = np.array([1, 2, 3, 4])
b = np.array([1, 2])

result = a + b
print("브로드캐스팅 결과:", result)
print("a의 shape:", a.shape)
print("b의 shape:", b.shape)
len(a)
len(b)

# b 배열을 반복 확장하여 a의 길이에 맞춤
b_repeated = np.tile(b, 2)
print("반복된 b 배열:", b_repeated)
# 브로드캐스팅을 사용한 배열 덧셈
result = a + b_repeated
print("브로드캐스팅 결과:", result)

a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b
a.shape
b.shape # 숫자 1개여서 shape이 존재하지 않음 

#차원 배열과 1차원 배열의 덧셈
import numpy as np
# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
                [30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector.shape
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

#열이 맞지 않을때 
#배열에 세로 벡터 더하기
matrix = np.array([[ 0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
                [30.0, 30.0, 30.0]])
matrix.shape
# 세로 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4) #1행 4열
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1) #4행 1열
vector

#브로드캐스팅이 되는 경우
(4,3) + (3,) = 가능
(4,3) + (4,) = 불가능
(4,3) + (4,1) = 가능 

# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)
