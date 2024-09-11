# Ctrl + Enter
# Shift + 화살표 : 블록
a=1
a

# 파워쉘 명령어 리스트
# ls: 파일 목록
# cd: 폴더 이동
# .  현재폴더
# .. 상위폴더

# Tab\Shift Tab 자동완성\옵션 변경

# Show folder in new window:
# 해당위치 탐색기

# cls: 화면 정리
a = 10
a
a = '"안녕하세요!" 라고 아빠가 말했다.'
a = '안녕하세요!'
a

a = [1 , 2, 3]
a

b = [4, 5, 6]
a + b

a = '안녕하세요!'
a

b = "4, 5, 6"
a + b

b = 'LS 빅데이터 스쿨!'
b

a + b
a + ' ' + b

a
print(a)

num1 = 3
num2 = 5
num1 + num2

a = 10
b = 3.3

a + b
a - b
a * b #곱하기기
a / b #나누기기
a // b # 정수 나눗셈 몫을 정수로 반환 
a % b  # 나머지 반환 
a ** b # 거듭제곱 

# Shift + Alt + 아래화살표: 아래로 복사
# Ctrl + Alt + 아래화살표: 커서 여러개
(a ** 3) // 7 
(a ** 3) % 7
(a ** 3) % 7
(a ** 3) % 7

#논리 연산자 비교 연산자 
# 불리언 값 조정 true / false 
a == b
a != b
a <= b
a >= b
a < b
a <= b


# 2에 4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지
# 9의 7승을 12로 나누고, 36452를 253로 나눈 나머지에 곱한 수
# 중 큰 것은?
a = ((2 ** 4) + (12453 // 7)) % 8
b = ((9 ** 7) / 12) * (36452 % 253)
a < b

user_age = 14
is_adult = user_age >= 18
print("성인입니까?", is_adult)

# False  = 3
# True = 2

a = "True"
type(a)
b = TRUE

c = true
d = True

# True, False
a = True
b = False

a and b
a or b
not a

# True: 1
# False: 0
True + True
True + False
False + False

# and 연산자
True and False
True and True
False and False
False and True

# and는 곱셉으로 치환가능
True  * False
True  * True
False * False
False * True

# or 연산자
True or True
True or False
False or True
False or False
TRUE = 4
a = False
b = False
a or b
min(a + b, 1)
max(a + b, 1)
sum(a + b, 1) 
# sum은 안댐 / iterable 객채여야 가능하다
#sum 함수는 일반적으로 interable객체를 더할때 사용된다
#iterable 객체 - 리스트 튜플 문자열 딕셔너리 집합 등 
a=[1,2,3]
b = [3,1,2]
sum(a)
sum([a+b],1) # 객체를 만들어주면 가능 


a = 100
a -= 20
a
a = 3
# a = a + 10
a += 10
a

a -= 4
a

a %= 3
a

a += 12
a **= 2
a /= 7
a


str1 = "hello"
strl + str1
str1 + " " + str1
str1 * 3
# 문자열 곱셉은 정수로만 가능

str1 = "Hello! "

# 문자열 반복
repeated_str = str1 * 3
print("Repeated string:", repeated_str)

str1 * -2


# 정수: int(eger)
# 실수: float (double)


# 단항 연산자
x=5
# 단항 덧셈 연산자 : 변수 값 변경 없이 원래 값(양수) 유지
# 값 강조할때 사용용
+x
# 단항 부정 연상자 :피연산자의 부호 반전/양수는 음수, 음수는 양수 
-x
# 비트 not 연산자 : 피연산자의 모든 비트 반전/ 0은 1로 1은 0으로 
~x

# binary

bin(-3)
bin(3)
bin(~5)

max(3, 4)
var1 = [1, 2, 3]
sum(var1)

!pip install pydataset

import pydataset
pydataset.data()

df = pydataset.data("AirPassengers")
df
