greeting = "안녕" + " " + "파이썬!"
print("결합된 문자열:", greeting)
type(greeting)

laugh = '하'*3
print('반복 문자열:',laugh)

# 리스트 생성 예제
 #다양한 데이터 타입/ 리스트 안에 리스트 
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]
print("Fruits:", fruits)
print("Numbers:", numbers)
print("Mixed List:", mixed_list)

# 튜플 생성 예제
a = (10, 20, 30) 
b_int=(42)
b_int
type(b_int)
b_int=10
b_int

b_tp = (42,)
b_tp
type(b_tp)

#튜플 인덱싱/슬라이싱

a_tp=(0,10,20,30,40,50)
a_tp[3:] # 해당 인덱스 이상 
a_tp[:3] # 해당 인덱스 미만 
a_tp[1:3] # 해당 인덱스 이상 & 미만 


a[0]
a_tp[1]=25 > 수정불가

#리스트 인덱싱/ 슬라이싱 

a_ls=[10,20,30,40,50]
#인덱싱
a_ls[0]
a_ls[1:4]
a_ls[:3]

#슬라이싱
a_ls[3:]
a_ls[:3]
a_ls[1:3]

a_ls[1]=25
a_ls

#튜플과 함수 
#사용자 정의 함수 
def min_max(numbers):
 return min(numbers), max(numbers)

#두값을 모두 튜플 형태로 반환 
result = min_max([1, 2, 3, 4, 5])
#> 리스트에서 구하기 

result = min_max((1, 2, 3, 4, 5))
#튜플에서 구하기 

print("Minimum and maximum:", result)

a=[1, 2, 3, 4, 5]
result = min_max(a)
#최소/최대값을 담은 튜플 변환  

result
result[0]=4
# 변경 불가(튜플이니까)
type(result)

#딕셔너리
#딕셔너리 생성 예제
#딕셔너리는 효율적인 데이터 검색, 수정 및 관리를 위한 다양한 부가기능(메서드)들을 제공
#주요 메서드로는 get(), update(), keys(), values(), items()

person = {
'name': 'John',
'age': 30,
'city': 'New York'
}

jaehee = {
  'name':'jaehee',
  'age':(27,25),
  'city':['미국','한국'] 
}
print('name',name)
print('jaehee', jaehee)

*get() 함수 활용해서 사용한 값 빼내오기 
jaehee.get('name')
jaehee.get('age')

#인덱싱 추가
jaehee.get('age')[0]

jaehee_age=jaehee.get('age')
jaehee_age
jaehee_age[0] 
#103랑 동일 

#집합형 데이터  set
#집합은 요소의 추가, 삭제 및 집합 간 연산을 지원.
#add(),remove(), update(), union(),intersection() 등의 메서드를 제공

fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits)
type(fruits)

empty_set = set()
print("Empty set:", empty_set)

empty_set.add('apple')
empty_set.add('apple')
empty_set.add('banana')
empty_set

empty_set.remove('banana')
empty_set.remove('cherry') > 에러 발생 ( 요소가 집합에 없음)
empty_set

empty_set.discard('banana') > 요소가 없어도 에러 발생 없음.
empty_set

#union(): 합집합 / #intersection(): 교집합

fruits = {'apple', 'banana', 'cherry', 'apple'}
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits)
union_fruits
intersection_fruits = fruits.intersection(other_fruits)
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)
intersection_fruits

#논리형 데이터타입 > T/F로만 값을 가진다 / T:1 F:0
#Boolean 타입에는 두 가지 값만: True (참)와 False (거짓).

p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p)

#논리형 값은 직접 할당 뿐 아니라 조건문의 결과로 생성되기도
is_active = True
is_greater = 10 > 5  # T 반환
is_equal = (10 == 5)  # F 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

age=10
is_active = True
is_greater = 10 > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

#if 문- 조건문에 활용 

a=1
if (a == 2):
 print("a는 2와 같습니다.")
else:
 print("a는 2와 같지 않습니다.")

a==2
 
# 데이터 타입 변환

#숫자형을 문자열형으로 변환 
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))
 
#문자열형을 숫자형(실수)로 변환 
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

#문자열형을 숫자형(정수)로 변환 
num_again = int(str_num)
print("숫자형:", num_again, type(num_again))

#리스트,튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

#집합, 딕셔너리 변환
set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

#논리형을 숫자형/ 문자형으로 바꾸기 

#논리형과 숫자형 변환 예제
#숫자형에서 논리형 
zero = 0
non_zero = 7
bool_from_zero = bool(zero) # False
bool_from_non_zero = bool(non_zero) # True
print("0를 논리형으로 바꾸면:", bool_from_zero)
print("7를 논리형으로 바꾸면:", bool_from_non_zero)

# 논리형을 숫자로 변환
true_bool = True
false_bool = False
int_from_true = int(true_bool) # 1
int_from_false = int(false_bool) # 0
print("True는 숫자로:", int_from_true)
print("False는 숫자로:", int_from_false)


# 논리형과 문자열형 변환 예제
# 논리형을 문자열로 변환
str_from_true = str(true_bool) # "True"
str_from_false = str(false_bool) # "False"
print("True는 문자열로:", str_from_true)
print("False는 문자열로:", str_from_false)

# 문자열을 논리형으로 변환
str_true = "True"
str_false = "False"
bool_from_str_true = bool(str_true) # True
bool_from_str_false = bool(str_false) # True, 비어있지 않으면 무조건 참
print("'True'는 논리형으로 바꾸면:", bool_from_str_true)
print("'False'는 논리형으로 바꾸면:", bool_from_str_false)
