fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]
 ## ch) 넘파이 어레이의 혼합은 자동으로 문자정보로 바꿔줌 

# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
range_list

range_list[3] = "LS 빅데이터 스쿨"
range_list

# 두번째 원소에 다음과 같은 정보를 넣어보자!
# ["1st", "2nd", "3rd"]
range_list[1] = ["1st", "2nd", "3rd"]
range_list

# "3rd" 만 가져오고 싶다면?
range_list[1][2]


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
exam = pd.read_csv("data/exam.csv")
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
repeated_list = [x for x in numbers for _ in range(4)]
repeated_list


# for 루프 문법
# for i in 범위:
#   작동방식
for x in [4, 1, 2, 3]:
    print(x)

for i in range(5):
    print(i**2)

# 리스트를 하나 만들어서
# for 루프를 사용해서 2, 4, 6, 8,..., 20의 수를
# 채워넣어보세요!
# mylist=list(range(1, 11))
# [i for i in range(2, 21, 2)]

mylist=[]
for i in range(1, 11):
    mylist.append(i*2)
mylists

# mylist는 비어있는 상태인데 i*2 가 돌아가면서 해당 값을 공백에 넣어줌 
# [].append[2] >>> [2].append(4) >>> [2,4].append(6) >>> [2,4,6] ...

mylist = [0] * 10
for i in range(10):
    mylist[i] = 2 * (i + 1)
mylist

# 요약
# 목적: 리스트를 생성하고, 각 인덱스에 2의 배수를 순차적으로 할당
# 단계:
## 01.길이가 10인 리스트를 0으로 초기화.
## 02.for 루프를 통해 리스트의 각 요소를 2 * (i + 1)로 설정.
## 결과: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]인 리스트가 생성.


# 인덱스 공유해서 카피하기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 10
for i in range(10):
    mylist[i] = mylist_b[i]
mylist


# 퀴즈: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에
# 가져오기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5
for i in range(5):
    mylist[i]=mylist_b[2*i]
    
mylist

# for i in range(5):각 반복에서 mylist의 i번째 요소를 mylist_b의 특정 요소로 설정.
# mylist[i]=mylist_b[2*i] : mylist_b의 인덱스를 2 * i로 계산하여 mylist의 i번째 위치에 할당.
# ex)
 ## i가 0일 때: mylist[0] = mylist_b[2 * 0]이므로 mylist[0] = mylist_b[0].
 ## i가 1일 때: mylist[1] = mylist_b[2 * 1]이므로 mylist[1] = mylist_b[2].
 ## 모든 반복이 끝나면, mylist의 각 요소는 mylist_b의 0, 2, 4, 6, 8번째 요소
 
--------------------------------------------------------------------
# 리스트 컴프리헨션으로 바꾸는 방법

# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서.
# for 루프의 : 는 생략한다.
# 실행부분을 먼저 써준다.
# 결과값을 발생하는 표현만 남겨두기

[i*2 for i in range(1, 11)]
[x for x in numbers]

for i in [0, 1]:
    for j in [4, 5, 6]:
        print(i, j)

# cf) np.repeat(numbers, 4, axis=None)
numbers = [5, 2, 3]
for i in numbers:
    for j in range(4):
        print(i)
        
numbers = [5, 2, 3]
for i in numbers:
    for j in range(4):
        print(i,j)

# 리스트 컴프리헨션 변환
[i for i in numbers for j in range(4)]

for x in numbers:
    for y in range(4):
        print(x)


# _ 의 의미
# 1. 앞에 나온 값을 가리킴
5 + 4
_ + 6    # _는 9를 의미

# 값 생략, 자리 차지(placeholder)
a, _, b = (1, 2, 4)
a; b
# _
# _ = None
# del _

## lopping ten times using _
for x in range(5):
    print(x)

--------------------------------------------------

# 원소 체크

fruits = ["apple", "banana", "cherry"]
fruits
"banana" in fruits

# [x == "banana" for x in fruits]
mylist=[]
for x in fruits:
    mylist.append(x == "banana")
mylist


# 바나나의 위치를 뱉어내게 하려면?

import numpy as np
fruits = ["apple", "apple", "banana", "cherry"]
fruits = np.array(fruits)  ##넘파이 배열은 파이썬 리스트와 유사하지만, 벡터화 연산과 같은 고급 기능 제공
int(np.where(fruits == "banana")[0][0])

# np.where(fruits == "banana")
# fruits 배열에서 "banana"와 일치하는 요소의 인덱스를 반환.
 ## fruits == "banana"는 불리언 배열T/F을 생성.
 
 ## 01. fruits == "banana"는 [False, False, True, False]가 됩니다.
 ## 02. np.where는 이 불리언 배열에서 True 값의 인덱스를 반환합니다.
 ## 03. np.where([False, False, True, False])는 (array([2]),)가 됩니다.
 ## 04. [0]은 첫 번째 반환 값을 선택. np.where는 튜플을 반환하는데, 첫 번째 요소가 우리가 원하는 인덱스 배열입니다.
 ## 05. np.where(fruits == "banana")[0]는 array([2])가 됩니다.
 ## 06. [0]은 이 배열의 첫 번째 요소를 선택하는 것입니다. array([2])[0]는 2

# 리스트 메서드 

# 원소 거꾸로 써주는 reverse()
fruits = ["apple", "apple", "banana", "cherry"]
fruits.reverse()
fruits

# 원소 맨끝에 붙여주기
fruits.append("pineapple")
fruits

# 원소 삽입 (바꾸기 x , 밀어 끼워넣기 )
fruits.insert(2, "test")
fruits

# 원소 제거
fruits.remove("test")
fruits

fruits.remove("apple")
fruits


import numpy as np

# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])

# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])

# 마스크(논리형 벡터) 생성
mask = ~np.isin(fruits, items_to_remove)
mask = ~np.isin(fruits, ["banana", "apple"])

#~np.isin(fruits, items_to_remove) - 바나나 /사과가 아닌 것들.

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)

#fruits[mask]
 ## mask 배열에서 True인 위치에 해당하는 fruits 배열의 요소를 선택.
 ## 예: fruits[[False, False, True, False, True]]는 ["cherry", "pineapple"]
 
 ### 결과적으로 filtered_fruits는 "banana"와 "apple"을 제외한 요소들로 이루어진 새로운 배열.
 
 
 
