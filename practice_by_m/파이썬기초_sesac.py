

# f 스트링
 # 사람 이름 리스트
names = ["June", "Alice", "Bob"]

# 반복문으로 개별 인사 출력
for x in names:
    print(f'{x}님, 안녕하세요')

# 인덱싱    
import numpy as np
a2 = np.array(["1반","2반","3반"])
np.random.seed(42)

a = np.arange(11)
a
b = np.arange(3).repeat(3)
b

b2 = b[[1,2,3]] 
b2

a2[b2==1] 

# 논리형 
a2 == "1반"

np.random.choice(a, size=10, replace=False) # 반복없이 
a[[num for num in range(4)]]
a[[b]]
a[[np.arange(3).repeat(3)]]
a[[np.where(b==1,7,0)]]
[num for num in a]

# 슬라이싱
a[4:]

# 문자열 다루기 =================

# 문자열 
# 양수 0,1,2,3 
# 음수 -3,-2,-1
a = "hello world"
a[3]
a[-1]
a[-3]
# 문자열슬라이싱
# 시작포함/ 끝미만
a[ 0 : 5] 
a[ : 5] # 0 생략 가능 

a[ 6 : 11]
a[-11 : -5]

a[-5 :]
a[::2] # 한 문자씩 건너띄워서 출력
a[::-1] # 역순 출력 

# 문자열 함수
# 문자열 교체
a = "l love strawberry"
a
a.replace("strawberry", "banana")

## 데이터 프레임에서 이름 변경 
df_exam = pd.read_excel("data/excel_exam.xlsx", 
                      sheet_name="Sheet2")
df_exam
df_exam.rename(columns={"nclass": "class"}, inplace=True)

# 데이터에서 replace 쓸때
# 예: nclass 열에서 값 'A'를 'B'로 변경
df_test = df_exam.copy()
df_test
df_test["class"] = df_test["class"].replace(1, "A")
df_test["class"] = df_test["class"].replace({1: "A", 2: "B"}) 
# 두개 이상은 딕셔너리

# 문자열 위치 찾기
a = "l love strawberry"
a
a.find('e')
a.find('r')

# 문자열 대소문자 변환
a_2 = 'hello,world'
a_2.lower()
a_2.upper()

# 문자열 나누기 
phone = "010-4196-9066"
phone.split('-')

email = "jenny0810@naver.com"
email.split('@')

full_name = input('영문이름(성과 이름은 공백으로 구분하세요):')
full_name

space = full_name.find(' ')
space

firt_name = full_name[ : space]
firt_name
last_name = full_name[space :]
last_name
# ===============================
# for 문
# 구구단 뱉어내기
# 가로 
for i in range(2,10):
    for j in range(1,10):
        print(f'{i}*{j} = {i*j}', end = "\t")
print() 

# 세로
for i in range(1,10):
    for j in range(2,10):
        print(f'{j}*{i} = {i*j}', end = "\t")
print() 

# 1부터 10까지 정수 누적합 구하기 
x = 0
for i in range(1,11):
    x += i
print(x)  

# 1부터 100까지 홀수 누적합 구하기 
x = 0
for i in range(1,100):
    if i%2 == 1:
        x += i
print(x) 

# 리스트 내포(comprehension)
# 1. 대괄호로 쌓여져있다 => 리스트다.
# 2. 넣고 싶은 수식표현을 x를 사용해서 표현
# 3. for .. in .. 을 사용해서 원소정보 제공
list(range(10))
squares = [x**2 for x in range(10)]
squares

# 3, 5, 2, 15의 3제곱
my_squares=[x**3 for x in [3, 5, 2, 15]]
my_squares

my_squares=[x**3 for x in [3, 5, 2, 15]]
my_squares

# Numpy 배열이 와도 가능
import numpy as np
my_squares=[x**3 for x in np.array([3, 5, 2, 15])]
my_squares

# Pandas 시리즈 와도 가능!
import pandas as pd
exam = pd.read_csv("../data/exam.csv")
my_squares=[x**3 for x in exam["math"]]
my_squares

# 리스트 합치기
3 + 2
"안녕" + "하세요"
"안녕" * 3
# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2

(list1 * 3) + (list2 * 5)

numbers = [5, 2, 3]
repeated_list = [x for x in numbers for y in range(4)]
repeated_list2 = [x for x in numbers for _ in range(4)]
repeated_list

# ===============================
# if 문
df_exam = pd.read_excel("data/excel_exam.xlsx", 
                      sheet_name="Sheet2")
df_exam   
df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"]
df_exam["mean"] = df_exam["total"] / 3
df_exam

df_exam[np.where(df_exam["english"] >=90, True , False)]
df_exam["eng_grade"] = np.where(df_exam["english"] >=90, "A" ,\
np.where(df_exam["english"] >=50,"B","C"))

if df_exam["english"] >=90 :
    eng_grade = "A"
elif 90 >= df_exam["english"] >=50:
    eng_grade = "B"
else:
    eng_grade = "C"

# ===============================
# for / while

for i in range (1,6) :
    print(i)

n = 1
while n <=5:
    print(n) # n을 먼저 출력
    n += 1   # 그 후 n을 1 증가 > 순서를 바꾸면 2부터 시작되버림

# ----------------
total = 0
for i in range( 1, 11) :
    total += i
print(total) # for문에 들여쓰면 단계별 누적값 다나옴

n =1
total = 0
while n<=10:
    total += n
    n += 1
print(total)

# while만 사용가능할때
# 'q'를 입력할때까지 반복하여 이름 받기
name = input ("이름:")
while name != "e":
    name = input("이름:")

# break로 빠져나오기 
 while True:
    name = input('이름:') 
    if name =="e":
        break

# 사용자가 원하는 조건을 입력할 때까지 숫자를 입력받아 
# 입력받은 숫자들의 합을 구하는 프로그램을 작성하시오

total = 0
while True :
    num = int(input("숫자:"))
    if num ==0:
        break
    total += num

print(total)

# 5.9.7.6.0 > 총 합 27

# 올바른 아이디/비번을 입력할때까지 아이디와 비번 입력하는 프로그램

id = "id123"
pwd = "pwd123"

while True:
    input_id = input('id:')
    input_pwd = input('pwd:')

    if id == input_id and pwd == input_pwd:
        break

#= ===================
# 알고리즘 연습하기

# up/down 숫자 맞추기 게임

# 정답숫자
import random
import numpy as np
num = random.randrange(1,101)
num #23

# 정답 맞출때까지 반복하기
# 정답을 맞추면 반복에서 벗어남
while True:
    answer = int(input("예상숫자:"))
    if answer == num:
        print("correct")
        break
    if answer < num:
        print("up")
    else:
        print('down')

# 게임 확장
# 기회는 5번/ 넘으면 '횟수초과'메세지 + 정답 제공 
# 정답 맞추면 몇번째에 맞추었는지 출력

# 횟수
count = 0
while True:
    count += 1
    if count > 5:
        print('횟수초과:정답은', num)
        break

    answer = int(input("예상숫자:"))
    if answer == num:
        print("correct")
        print(count,'번만에 맞추었습니다.')
        break
    if answer < num:
        print("up")
    else:
        print('down') 

# 사칙연산 프로그램

# 두 수와 사칙연산기호를 입력받아 연산 기호에 따라 연산 결과를 출력
# 사칙연산기호가 아닌 경우 '잘못입력하셨습니다' 출력
num1 = int(input("숫자1:"))
num2 = int(input("숫자2:"))
op = input('연산기호:')

if op == "+" :
    print(f'{num1} + {num2} = {num1 + num2}')
elif op =="-":
    print(f'{num1} - {num2} = {num1 + num2}')   
elif op =="*":
    print(f'{num1} * {num2} = {num1 + num2}')   
elif op =="/":
    print(f'{num1} / {num2} = {num1 + num2}')
else :
    print('not correct word!') 

# 할인된 금액 계산
# 물건 구매가를 입력받고, 금액에 따른 할인율 계산
# 구매가 /할인율 /할인금액 /지불금액   출력

price = int(input("물건 구매가:"))

if price >= 100000:
    dc = 10
elif price >= 50000:
    dc = 7
elif price >= 10000:
    dc = 5
else:
    dc = 0

print(f'''

구매가:{price}
할인율:{dc}
할인금액:{price * (dc / 100)}
지불금액:{price - (price * (dc / 100))}

''')







