#균일 확률 변수
import numpy as np 

np.random.rand(1)

#함수정의
def X(num):
    return np.random.rand(num)

X(1)
X(2)
X(3) 
#한꺼번에 3개 뽑는게 아니라 하나씩 뽑은것을 세개 보여주는 것 
#똑같은 값이 같이 나올 수 있음 


X(1)
#베르누이 확률변수 모수:P 만들어보세요
def Y(p):
    x=np.random.rand(1)
    return np.where(x < p, 1,0)
Y(0.5)

def Y(num, p):
    x=np.random.rand(num)
    return np.where(x < p, 1, 0)

Y(num=5, p=0.5)
sum(Y(num=100, p=0.5)) / 100
Y(num=100, p=0.5).mean() # 평균값은 계속 달라짐
Y(num=10000, p=0.5).mean()


#새로운 확률변수 
#가질 수 있는 값: 0,1,2 ( 중 하나가 나와야함 )
#20% 50% 30%


def Z():
    x=np.random.rand(1)
    return np.where(x < 0.2, 0, np.where(x < 0.7, 1, 2))
        
Z()

# p= np.array([0.2, 0.5, 0.3])
def Z(p):
    x=np.random.rand(1)
    p_cumsum=p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1], 1, 2))

p = np.array([0.2, 0.3, 0.5])        
Z(p)

import numpy as np 

#기대값 E(X)
#0 * 0.2 +1*0.5+ 2*0.3
#R.V 가질수 있는 값 X에 대응하는 확률 

#E(X)
x=0,1,2,3
p(x) = 1/6, 2/6, 2/6 ,1/6
sum(np.arange(4) * np.array([1,2,2,1])/6)



