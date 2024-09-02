
# 이삭샘 교재 10TH

def g(x=3):
 result = x + 1
 return result
g()

# 함수 내용 확인 
import inspect
print(inspect.getsource(g))

import numpy as np
np.array([1, 2, 3])

x = 3
if x > 4:
   y = 1
else:
   y = 2
print(y)   

# if else 축약 
y = 1 if x > 4 else 2 
y

# 리스트 컴프리헨션
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]
print(result)

x = 0
if x > 0:
    result = "양수"
elif x == 0:
    result = "0"
else:
    result = "음수"
print(result)


# 조건 3가지 넘파이 버전 
import numpy as np

x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0, x == 0, x < 0]
choices = ["양수", "0", "음수"] 
result = np.select(conditions, choices)
print(result)


# for loop 예시
for i in range(1, 4):
    print(f"Here is {i}")
# for loop 리스트 컴프
[f"Here is {i}" for i in range(1, 4)]

name = "John"
age = 30
greeting = f"Hello, my name is {name} and I am {age} years old."
print(greeting)    


import numpy as np
names = ["John", "Alice"]
ages = np.array([25, 30]) # 나이 배열의 길이를 names 리스트와 맞춤

# 각 이름과 나이에 대해 별도로 인사말 생성
greetings = [f"Hello, my name is {name} and I am {age} years old." for 
name, age in zip(names, ages)]
for greeting in greetings:
    print(greeting)

# while, break문
i = 0
while True:
    i += 3
    if i > 10:
        break
    print(i)


import pandas as pd
data = {'A': [1, 2, 3], 
        'B': [4, 5, 6]}

df = pd.DataFrame(data)
print(df)

df.apply(max, axis=0)
df.apply(max, axis=1)

df.apply(sum, axis=0)
df.apply(sum, axis=1)


def my_func(x, const=3):
    return max(x)**2 + const

my_func([3,4,10], 5)

df.apply(my_func, axis=1, const=5)


import numpy as np
array_2d = np.arange(1, 13).reshape((3, 4), order='F')
array_2d

np.apply_along_axis(max, axis=0, arr=array_2d)

# 함수 환경

y = 2

def my_func(x):
    global y

    def my_f(x):
        return x ** 2
    
    y = my_f(x) + 1
    result = x + y
    return result

my_func(2)
print(y)

# 입력값이 몇 개일지 모를땐 별표를 * 붙임 
def add_many(*args):
    result = 0
    for i in args:
        result = result + i 
    return result
    
add_many(1, 2, 3)



def first_many(*args):
    return args[0]

first_many(1, 2, 3)
first_many(4, 1, 2, 3)


def add_mul(choice, *args):
    if choice == "add":
        result = 0
        for i in args:
            result += i
    elif choice == "mul":
        result = 1
        for i in args:
            result *= i
    return result

add_mul("mul", 5, 4, 3, 1)

## 별표 두개 (**) 는 입력값을 딕셔너리로 만들어줌! 
def my_twostars(choice, **kwargs):
    if choice == "first":
        return kwargs[0]
    elif choice == "second" :
        return kwargs[1]
    else:
        return kwargs
    
my_twostars("first", age = 30, name = "issac")   

dict_a = {"age" : 30, "name" : 'issac'}
dict_a["age"]












