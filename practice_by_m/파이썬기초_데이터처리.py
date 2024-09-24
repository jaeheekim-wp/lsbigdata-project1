# 리스트
l1 = []
l1
l2 = list()# dddd
l2
l3 = [1,3,5]
l3
l4 = list(range(1,11,2))
l4 
# 그냥 리스트 

import numpy as np
l4_1 = np.arange(1,11)
l4_1
# 넘파이 어레이

# 튜플 
# 프로그램 실행되는 동안 값이 바뀌면 안될때
# 함수에 인수를 전달하거나 값을 리턴할때 사용되는 경우 많다

t1 = ()
t1
t2 = tuple()
t2
t3 = (1,3,5,7,9)
t3
t4 = tuple(range(1,100,2))
t4

t5 = 1,3,5,7,9 # 괄호 생략 가능 
t5

t6 = (1,)# 항목이 1개일때는 , 필수
t7 = (1)

type(t6)
type(t7)


# 리스트/튜플의 자료형
# 모든 자료형이 혼합되어 들어갈 수 있다.
myinfo_l = ["jaehee",27, 163.5, ['drawing','reading']]
type(myinfo_l)
myinfo_t = ("jaehee",27, 163.5, ['drawing','reading'])
type(myinfo_t)

l1 = ["a","b","c"]
l2 = ["d", "e"]
l1 + l2
l1 * 2
"a" in l1
"Y" in l1
"a" not in l1
len(l1)
l3 = ["jaehee",27, 163.5, ['drawing','reading']]  
len(l3)

t1 = ("a","b")
t2 = ("c", "d")
t1 + t2
t1 * 2
"a" in t1
"Y" in t1
"a" not in t1


# 사용자 정의 함수
# 특정 기능 구현하기 위한 <코드 묶음>
# 내장함수/외장함수/사용자정의함수

# 함수 정의하기
def introduce():
    print('안녕하세요!')
    print('제 이름은 파이썬입니다.') 

introduce()

# def 함수명(매개변수1, 매개변수2,...) : 
# 함수내용

name = '김재희'
print('안녕하세요')
print('저의 이름은,'+name+'입니다.')

def introduce(name) :
    print('안녕하세요')
    print('저의 이름은,'+name+'입니다.')

 introduce("재희")
 introduce('jaehee')

def introduce2(name, age):
    print('안녕하세요')
    print('저의 이름은,'+name+'입니다.')
    print('나이는,'+str(age)+'살입니다.')

 introduce2("jaehee",27)  

def number(i):
    for i in range(2,11):
        print(f'제 번호는 {i} 입니다')
number(3)


# 연습문제
# 이름과 나이를 입력받아 생일 축하 메세지를 출력하는 함수
# 생일 문구 00님의 00번째 생일을 축하합니다

def birthday(name, age):
    print(''+name+'님의 '+str(age)+'번째 생일을 축하합니다!')

def birthday2(name, age):
    print(f'{name}님의 {age}번째 생일을 축하합니다!')

birthday("jaehee", 27)
birthday2("june", 29)

# ======input 활용=========
name = input('name:')
age = input("age:")
def birthday2(name, age):
    print(f'{name}님의 {age}번째 생일을 축하합니다!')
birthday2(name, age)
# ======input 활용=========


# 함수 결과값 받아서 사용하기

# def 함수명(매개변수1, 매개변수2,...) : 
# 함수내용
 ## return 결과값


# 1개 반환
def get_plus(n1,n2):
    return n1 + n2

get_plus(2,4)
plus = get_plus(2,4)
print("pluse:", plus)

# 여러개 반환 - 하나의 튜플로 묶어서 반환해줌
def get_plus_mius(n1,n2): # (n1,n2) 위치인수
    return n1 + n2 , n1 - n2
get_plus_mius(2,4)

plus, minus = get_plus_mius(4,4)
plus
minus

# return 을 만나면 함수에서 빠져나옴/ 
# 반환값이 잇으면 값 반환, 없으면 그냥 빠짐

# 입력값:정수/ 출력:0,짝수,홀수 리턴하기
def is_odd_even(n):
    if n == 0 :
        result = 0
    elif n % 2 == 0 :
        result = "짝수"
    else:
        result = "홀수"
    return result

is_odd_even(10)
is_odd_even(0)
is_odd_even(7)

## 사이에 return 넣어서 정리하기
def is_odd_even(n):
    if n == 0 :
        return 0
    elif n % 2 == 0 :
        return "짝수"
    return '홀수'

is_odd_even(0)
is_odd_even(8)

# 연습문제
# 소수 여부를 판단하기 
# 매개변수로 전달받은 수가 소수인지 아닌지 판별하는 함수 작성하고 호출
# 소수란 1과 자기 자신으로만 나누어 떨어지는 1보다 큰 양의 정수

def is_prime(n):
    if n <= 1 :
        return False
    for i in range(2,n):
        if n % i == 0:
            return False
    return True

is_prime(2)
is_prime(3)
is_prime(15)

# ==========================
# 디폴트인수 (ch. 위치인수)
# 디폴트 값을 지정하면 디폴트값이 있는 인수를 생략할 수 있음 
# 기본 위치인수(ex.name) 을 다 적은 다음에 적어야 함 
def greet(name, msg = 'nice to meet you!'):
    print ('hello',name,msg)

greet("jaehee")
# 디폴트 인수에 값 지정하면 지정한 값으로 출력 
greet("jaehee", "long time no see!") 

# ==========================
# 키워드 인수 
# 함수를 호출할때 인수의 이름을 명시하면, 순서를 바꾸어 전달할 수 있다
def get_minus(x,y,z):
    return x-y-z

get_minus(10,5,10)
# 키워드 인수는 기본 위치인수를 다 적은 다음에 적어야함 
get_minus(10,z=10, y=5,)

# ==========================
# 가변인수
# 인수를 하나의 튜플이나 리스트로 전달
def average(args) :
    return sum (args) / len (args)

average([1,2,3])
average((1,2,3,4,5))

# 매개변수에 *를 붙이면 여러개의 인수를 하나의 튜플로 받는다
def average(*args):
    print(args)
    return sum (args) / len (args) 

average(1,2,3)


# print 함수의 위치인수,키워드인수
help(print)

print(1, 2, 3, sep='@')
print(4,5)







