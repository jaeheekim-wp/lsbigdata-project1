

fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]
print("과일 리스트:", fruits)
print("숫자 리스트:", numbers)
print("혼합 리스트:", mixed)
 ## ch) 넘파이 어레이의 혼합은 자동으로 문자정보로 바꿔줌 
 
empty_list1 = []
empty_list2 = list()
print("빈 리스트 1:", empty_list1)
print("빈 리스트 2:", empty_list2)


numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))

print("숫자 리스트:", numbers)
print("range() 함수로 생성한 리스트:", range_list)


range_list[3]
range_list[3] = "빅데이터스쿨"

두번째 원소에 
range_list[1] =  ["1st","2nd","3rd"]
range_list

# 3rd만 가져오기
range_list[1][2]

# 리스트 내포 comprehension

# 대괄호로 쌓여져잇음 -- > 리스트다
# 넣고 싶은 수식표현을 x를 사용해서 표현
# for..in..을 사용해서 원소정보 제공 

list(range(10))
squares = [x**2 for x in range(10)] # 제곱리스트 
squares 

# 3,5,2,15의 3 제곱곱
my_squares = [x**3 for x in [3,5,2,15]]
my_squares


# np 어레이로도 가능
import numpy as np 
np.array([3,5,2,15])
my_squares = [x**3 for x in np.array([3,5,2,15])] 
my_squares


import pandas as pd
# pandas시리즈도 가능
exam = pd.read_csv("data/exam.csv")  

my_squares = [x**3 for x in exam["math"]] 
my_squares


# 리스트 합치기 - 문자열과 비슷하게 붙여준다 

"안녕" + "하세요" # 붙이기 
"안녕" * 3 # 반복 
3+2
3*5

list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2

(list1 * 3) + (list2 * 5) 

# 각 원소를 반복하기
# [표현식 for 항목 in 반복 가능한 객체]

numbers = [5, 2, 3]
repeated_list = [x for x in numbers for _ in range(3)] # 앞에 나온 값 지칭 가능성
repeated_list

# 중첩된 for 루프를 사용하여 각 원소를 여러 번 반복합니다. 
# [x for x in numbers for _ in range(3)]
# for x in numbers는 numbers리스트의 각 항목을 반복
# 내부의 for _ in range(3)` 루프는 각 항목에 대해 3번 반복.


repeated_list = [x for x in numbers for _ in range[4,1,2,3]]
repeated_list

# - 의 의미
# 앞에 나온 값을 가리킴

5+4
_ + 6 # _는 9를 의미

del _

# 값 생략, 자리 차지(placeholder)
a, _, b = (1,2,4)
a; b 
_

# for 루프 문법
# for i in 범위:
#   작동방식

for x in [4,1,2,3]:
    print(x)
    
for i in range(5):
    print(i**2)
    
    
# 리스트를 하나 만들어서 
# for 루프를 사용해서 2,4,6,...,20의 수를 채워넣어보세요

[i for i in range(2,21,2)]  # 맨 앞에 i가 또 오는 이유 

mylist = []
for i in range(1,11) :
    mylist.append(i*2)
mylist
#mylist는 비어있는 상태인데 i*2 가 돌아가면서 해당 값을 공백에 넣어줌 
#[].append[2] >>> [2].append(4) >>> [2,4].append(6) >>> [2,4,6] ...

mylist = [0] * 10
for i in range(10):
    mylist[i] = 2 * (i +1)  # mylsit[i] ??

# 인덱스 공유해서 카피 
mylist_b = [2,4,6,8,10,12,14,16,18]





# 퀴즈 : mylist_b의 홀수번째 '위치'에 있는 숫자들만 mylist에 가져오기 
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5



# 리스트 컴프리헨션으로 바꾸는 법

# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서
# for 루프의 : 는 생략한다
# 실행부분을 먼저 써준다
# 결과값을 발생하는 표현만 남겨두기 
[i*2 for i in range(1, 11)]
[x for x in number]

for i in [0,1,2]:
    for j in [0, 1]:
        print(i,j)

number = [5,2,3]
for i in numbers:
    for j in range(4):
        print(i)
        
# 컴프리헨션
[i for i in numbers for j in range(4)]

# 원소 체크
fruits = ["apple", "banana", "cherry"]
fruits
"banana" in fruits

[x == "banana" for x in fruits]
for x in fruits :
    x == "banana"
    
mylist = []
for x in fruits:
    mylist.append(x =="banana")
mylist


바나나 위치 추출출
fruits = ["apple","apple", "banana", "cherry"]

import numpy as np 
fruits = np.array(fruits)
np.where(fruits == "banana")
np.where(fruits == "banana")[0] # 넘파이 어레이 행렬
np.where(fruits == "banana")[0][0] 
int(np.where(fruits == "banana")[0][0])

# 원소 반대로
fruits = ["apple","apple", "banana", "cherry"]
fruits.reverse()
fruits


# 원소 맨 끝에 추가
fruits.append("pineapple")
fruits

# 원소 삽입 "밀고 끼워넣기"
fruits.insert(2, "test")
fruits


fruits.remove("test")
fruits
fruits.remove("apple")
fruits





import numpy as np

# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])

# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])

# 불리언 마스크(논리형 백터) 생성
mask = ~np.isin(fruits, items_to_remove)
mask = ~np.isin(fruits, ["banana", "apple"]) # 리스트로 와도 가능.

#~np.isin(fruits, items_to_remove) - 바나나 /사과가 아닌 것들.

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)





