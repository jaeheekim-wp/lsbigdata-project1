# 데이터 타입
x = 15.34
print(x, "는 ", type(x), "형식입니다.", sep='')


# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

print(a, type(a))
print(b, type(b))


# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""

print(ml_str)
print(ml_str, type(ml_str))


# 문자열 결합
greeting = "안녕" + " " + "파이썬!"
print("결합 된 문자열:", greeting)

# 문자열 반복( 정수int로만 가능)
laugh = "하" * 3
print("반복 문자열:", laugh)

# 리스트
# 순서가 있고 수정가능
# 대괄호[] 사용 , 다양한 메서드 통해 데이터 관리 , 인덱싱/슬라이싱 
fruit = ["apple", "banana", "cherry"]
type(fruit)

numbers = [1, 2, 3, 4, 5]
type(numbers)
numbers = [1, 2, 2, 3, 4, 5]
type(numbers)

mixed_list = [1, "Hello", [1, 2, 3]]

type(mixed_list)

#인덱싱/ 슬라이싱 [이상:미만]
a_ls=[10, 20, 30, 40, 50]
a_ls[1:4]
a_ls[:3]
a_ls[2:]

# 해당 인덱스 변경 
a_ls[1]=25
a_ls

# 튜플
# 한번 생성된 후 수정 불가 
# 소괄호() 또는 괄호 없이 쉼표 구분해 사용 , 중복요소 포함 가능 

a_tp = (10, 20, 30, 40, 50) # a = 10, 20, 30 과 동일
a_tp[3:] # 해당 인덱스 이상
a_tp[:3] # 해당 인덱스 미만
a_tp[1:3] # 해당 인덱스 이상 & 미만

a_tp[1]
a[1] = 25 # 수정 불가 

b_int = (42)
b_int
type(b_int)
b_int = 10
b_int

# 요소가 하나인 튜플을 만들때는 요소 뒤에 쉼표 붙여서 구분 
b_tp = (42,)
b_tp
type(b_tp)

# 사용자 정의함수
def min_max(numbers):
  return min(numbers), max(numbers)

a=[1, 2, 3, 4, 5]
result = min_max(a)
result
type(result) # 튜플로 반환 
result[0]
result[0] = 4 # 따라서 수정 불가 

print("Minimum and maximum:", result)

# 딕셔너리
# 중괄호{} 사용
# key : value 의 쌍으로 데이터 저장 , 데이터 추가 삭제 수정 자유로움
# key는 변경 불가능- 보통 문자열,숫자,튜플 활용 

person = {
 'name': 'John',
 'age': 30,
 'city': 'New York'
}

print("Person:", person)

# 딕셔너리를 데이터프레임으로 만들기 ( pandas라이브러리 )
# value 리스트로 있어야 

import pandas as pd

person = {
 'name':["john", "bob", "david"],
 'age': [30, 20, 50],
 'city': ['New York',"houston","chicago"]
}

person = pd.DataFrame(person)
person

issac = {
  "name": "이삭",
  "나이": (39, 30),
  "사는곳": ["미국", "한국"]
}

#딕셔너리 메서드:get(),update(),keys(),values(),items()

print("Issac:", issac)

issac.get('나이')
issac.values()
issac.keys()

issac_age=issac.get('사는곳')
issac_age
issac_age[0]

# 집합 set
# 중괄호{} 또는 set() 생성자 활용 
# 순서가 없음, 중복 요소 허용 불가 
# 수학적인 집합 연산 지원/ 중복 제거 /항목 존재 여부 검사 등 
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨
type(fruits)


# 빈 집합 생성- set()
empty_set = set()
print("Empty set:", empty_set)

# 집합 메서드 : add(), remove(), update() , union(), intersection() 등 

empty_set.add("apple")
empty_set.add("banana")
empty_set.add("체리")

empty_set.add("apple")
empty_set.remove("banana") # 요소가 집합에 없으면 에러 발생 
empty_set.discard("banana") # 요소가 집합에 없어도 에러 발생 없음 
empty_set.pop()
empty_set.clear()
empty_set

# 집합 간 연산
fruits = {'apple', 'banana', 'cherry', 'apple'}
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits)  # 합집합( 중복 제거 )
intersection_fruits = fruits.intersection(other_fruits) #교집합
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)


# 논리형 데이터 예제
# True (1) , False(0) 만을 값으로 가짐 
# 변수에 직접 할당 가능 
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) 

age = 10
is_active = True
is_greater =  age > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

# 조건문
a=10
if (a > 5):
  print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")


#데이터 타입 변환 
# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))
num_again = int(str_num)
print("숫자형:", num_again, type(num_again))


# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

# 집합을 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
set_example = {'a', 'b', 'c'}
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

#논리형과 숫자형 
bool_t = True
bool_f = False
to_int_t = int(bool_t)
to_int_t

zero = 0
non_zero = 5
to_bool_zero = bool(zero)
to_bool_non_xero = bool(non_zero)

#문자열을 논리형으로 
str_true = "True"
str_false = "False"
bool_t_s = bool(str_true)
bool_f_s = bool(str_false)
