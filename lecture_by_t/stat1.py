# 통계

# 균일확률변수 만들기 > 확률 밀도함수가 그려진다 
import numpy as np

# np.random.rand() 
# 스탑 버튼/ 공을 꺼내는
# 0 이상 1 미만 사이의 랜덤한 실수(float)를 생성
# 이를 통해 난수 배열을 만들 수 있음.

# np.random.rand() #스탑 버튼/ 공을 꺼내는

 ## 0과 1 사이의 실수 생성: 기본적으로 0 이상 1 미만의 난수를 생성.

 ## 배열 형태 지정: 원하는 형태(차원)의 배열을 생성.
  ### 난수(亂數, Random Number)
  ### 예측할 수 없고 일정한 패턴이 없는 수. 즉, 무작위로 생성된 수
  
# 단일 난수
random_number = np.random.rand()
print(random_number)

# 다수 난수 
##1차원
random_numbers = np.random.rand(5)
print(random_numbers)

##2차원
random_matrix = np.random.rand(3, 2)
print(random_matrix)

##다차원
random_tensor = np.random.rand(2, 3, 4)
print(random_tensor)


# np.random.randint()
# 특정 범위 내에서 랜덤한 정수(int)를 생성
# 최소값과 최대값(최대값은 포함되지 않음)을 지정할 수 있음.
# 입력한 차원 크기만큼의 배열을 생성할 수 있음.


# 파이썬에서 난수 생성 예제

## 0과1사이의 난수 생성 
random_number = np.random.rand()
print(random_number)

## 1부터 10 사이의 정수 난수 생성
random_integer = np.random.randint(1, 11)
print(random_integer)

# 주요 차이점
# 생성되는 값의 유형:
# np.random.rand(): 0과 1 사이의 실수(float) 생성.
# np.random.randint(): 지정한 범위 내의 정수(int) 생성.

# 범위:
# np.random.rand(): 항상 0 이상 1 미만의 실수.
# np.random.randint(): 사용자가 지정한 최소값 이상, 최대값 미만의 정수.

# 사용 예시:
# np.random.rand(): 실수형 데이터를 필요로 할 때 사용.
# np.random.randint(): 정수형 데이터를 필요로 할 때 사용.

#활용 예제

##게임 개발: 캐릭터가 적을 공격할 때, 무작위로 공격 데미지를 결정.
##통계학: 무작위 표본을 추출하여 데이터 분석을 수행.
##암호화: 보안 키를 생성할 때 무작위 수를 사용하여 보안 강화.


# 베르누이 확률 변수

# 아주 간단한 형태의 확률변수

# 오직 '두 가지 결과'만을 가질 수 있는 상황을 모델링.

# 오직 두 가지 결과만을 가질 수 있는 상황을 모델링.

# 이 두 가지 결과는 보통 "성공"과 "실패" 또는 "1"과 "0"으로 표현.

#기본 개념
 ##확률: 각 결과가 나올 확률을 지정합니다. 
    ### ex)동전 던지기-앞면이 나올 확률이 0.5(50%), 뒷면이 나올 확률도 0.5(50%)

 ##값: 베르누이 확률변수는 두 가지 값 중 하나를 가집니다. 
    ### ex)"성공"일 때는 1, "실패"일 때는 0.


# 베르누이 확률변수의 성질

#베르누이 확률변수의 성질


 ## 성공 확률(p): 어떤 사건이 "성공"할 확률을 나타냅니다. 
    ### ex) p = 0.5라면 성공 확률이 50%입니다.
    
 ## 실패 확률(1 - p): "실패"할 확률입니다. 
    ### 성공 확률이 p라면, 실패 확률은 1 - p입니다.

np.random.rand(1)

def X(num):
    return np.random.rand(num)

X(1)
X(4)

# 베르누이 확률변수 모수: p 만들어보세요!
# num = 3
# p = 0.5

def Y(num, p):
    x=np.random.rand(num)
    return np.where(x < p, 1, 0)

# sum(Y(num=100, p=0.5))/100
Y(num=1, p=0.5)
Y(num=100, p=0.5).mean()

# 100개의 난수 발행, 그 값들은 0이상 1미만, 그 값들의 평균값
Y(num=100000, p=0.5).mean()
# 평균 값은  0.5 언저리에 있음.


# 새로운 확률변수 X
# 가질수있는 값 x : 0, 1, 2
# 확률 P : 20%, 50%, 30%


Y(num=100000, p=0.5).mean()

# 새로운 확률변수
# 가질수있는 값: 0, 1, 2
# 20%, 50%, 30%

def Z():
    x=np.random.rand(1)
    return np.where(x < 0.2, 0, np.where(x < 0.7, 1, 2))

Z()

# np.cumsum() 함수 활용 (배열의 누적 합을 계산)
# 확률 P = np.array([0.2, 0.5, 0.3])

# p = np.array([0.2, 0.5, 0.3)
def Z(p):
    x=np.random.rand(1)
    p_cumsum=p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1], 1, 2))

p = np.array([0.2, 0.5, 0.3])
p = np.array([0.1, 0.6, 0.3])


p = np.array([0.1, 0.6, 0.3])

Z(p)
