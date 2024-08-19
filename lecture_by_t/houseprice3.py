# 회귀분석

# 두 개 이상의 변수 사이의 관계를 분석하고, 
# 이를 통해 미래의 값을 예측하거나 변수들 간의 영향을 이해하려는 통계적 기법
# 회귀분석을 수행하는 주된 이유는 다음과 같습니다:
# 관계 파악/ 예측 / 원인 결과 이해 / 최적화 / 평가와 검증


# 직선의 방정식 
# y = ax +b 
# ex) y = 2X+3
a = 1
b = 3

x = np.linspace(-5,5,100)
y = a * x + b
plt.plot(x, y, color="#E85F33")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.show()
plt.clf()

---------------
# y = x
a = 1
b = 3

x = np.linspace(-5,5,100)
y = x
plt.plot(x, y, color="red")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.show()
plt.clf()


# b : 절편은 들어올리는 기능 'y 절편'
# a : (곱해진 수) 는 기울기 조절

---------------

a = 50
b = 50

x = np.linspace(0, 5, 100)
y = a * x + b

# 트레인 집 정보 가져오기
house_train=pd.read_csv("./data/houseprice/train.csv")

# 방갯수 데이터 
my_df=house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df
# 집값 1/1000로 보기 
my_df["SalePrice"]=my_df["SalePrice"] /1000
# 그래프 
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"])
plt.plot(x, y, color="blue")
plt.show()
plt.clf()


# houseprice
# 테스트 집 정보 가져오기
house_test = pd.read_csv("./data/houseprice/test.csv")
a=70
b=10
(a * house_test["BedroomAbvGr"] + b) * 1000

# sub 데이터 불러오기
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# SalePrice 바꿔치기
sub_df["SalePrice"] = (a * house_test["BedroomAbvGr"] + b) * 1000
sub_df


# csv s내보내기
sub_df.to_csv("./data/houseprice/sample_submission3.csv", index=False)

# =========================

# 직선 성능 평가
a=16
b=117

# y_hat 어떻게 구할까?
y_hat=(a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디에 있는가?
y = house_train["SalePrice"]

np.abs(y - y_hat)  # 절대거리
np.sum(np.abs(y - y_hat)) # 절대값 합
np.sum((y - y_hat)**2) # 제곱합

# 1조: 106021410
# 2조: 94512422
# 3조: 93754868 1등
# 4조: 81472836
# 5조: 103493158

# 1조: 106021410
# 2조: 94512422
# 3조: 93754868 
# 4조: 81472836 1등
# 5조: 103493158
# 9459298301338
# 회귀(절대값): 79617742
# 회귀(제곱합): 82373710


# ====================================
# 선형 회귀 모델 

# !pip install scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
x
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)
y

# 선형 회귀 모델 생성 (제곱방식)
model = LinearRegression()

# 모델 학습
 ## 자동으로 기울기, 절편 값을 구해줌
model.fit(x, y) 

# 회귀 직선의 기울기와 절편
 ## 값을 보여줌
model.coef_      # 기울기 a / 튜플 안 리스트 
model.intercept_ # 절편 b
type(model.coef_)
type(model.intercept_)
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)
y_pred

# 데이터와 회귀 직선 시각화
import matplotlib.pyplot as plt

# 데이터 시각화
plt.scatter(x, y, color='#F1D3CE', label='data')
plt.plot(x, y_pred, color='#D1E9F6', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# 범례 박스
legend = plt.legend(facecolor='#D1E9F6', fancybox=True)

# 마진 조정 (왼쪽, 오른쪽, 아래, 위)
plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.9)

# 그래프 전체 배경 색상 설정
plt.gcf().set_facecolor('#F6EACB')

# 그래프 안쪽 배경 색상 설정
plt.gca().set_facecolor('#F6EACB')  # 안쪽 배경 색상 (예: 흰색)
plt.show()
plt.clf()


# ============================
# houseprice

## 01. 회귀모델을 통한 집값 예측
 ## "방 갯수가 늘어날수록 집값이 얼마나 증가하는지"를 계산하는 도구
 ##  이 도구가 바로 선형 회귀 모델: model = LinearRegression()

# 필요한 패키지 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## 회귀분석 적합(fit)하기 
x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1)
y = house_train["SalePrice"]/1000

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습 
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산( x에 대한 y^값을 알려줌)
y_pred = model.predict(x)     

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

## 02. kaggle submission

test_x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1) 
test_x

pred_y = model.predict(test_x) # test셋에 대한 집값 
pred_y
len(pred_y)

# SalePrice 바꿔치기 - 계속 안댐 

#sub_df = pd.DataFrame({'Id' : house_test['Id'],
                       # 'SalePrice' : test['Id']})
sub_df['SalePrice'] = pred_y*1000
sub_df

# csv파일 내보내기 
sub_df.to_csv("./data/houseprice/sample_submission4.csv", index=False)

# ====================================

# 조별 회기분석 예측 
# 원하는 변수 사용해서 회귀모델을 만들고 제출할 것.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#필요한 데이터 불러오기
house_test = pd.read_csv('data/houseprice/test.csv')
house_train = pd.read_csv('data/houseprice/train.csv')
sub_df = pd.read_csv('data/houseprice/sample_submission.csv')

#이상치 탐색 및 제거 
#house_train.query("GrLivArea > 4500") #탐색
#house_train = house_train.query("GrLivArea <= 4500") #4500보다 작거나 같은 것만 할당해줌
 ## house_train['GrLivArea'].sort_values(ascending = False).head(2)

# 회귀분석 적합(FIT)하기 
x = np.array(house_train["GrLivArea"]).reshape(-1, 1)
# x = (house_train[["GrLivArea"]]) 로도 가능 > 데이터 프레임으로 
y = np.array(house_train["SalePrice"]) / 1000 

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) #fit함수가 자동으로 기울기, 절편 값을 구해줌.

# 회귀 직선의 기울기와 절편
model.coef_         #기울기 a
model.intercept_    #절편 b


slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산 
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='#FFC96F', label='data')
plt.plot(x, y_pred, color='#ACD793', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900]) 

# 범례 박스
# 범례 추가 및 사이즈 조정
legend = plt.legend(
    loc='upper left',          # 범례 위치
    bbox_to_anchor=(0, 1),      # 범례 박스의 앵커 위치 (좌표 1,1)
    fontsize='small',           # 폰트 크기 설정 ('small', 'medium', 'large' 등 사용 가능)
    title='Legend',             # 범례 제목 추가
    title_fontsize='5',        # 범례 제목 폰트 크기
    fancybox=True,              # 둥근 모서리 설정
    shadow=False,                # 그림자 추가
    framealpha=0.2,             # 투명도 설정
    facecolor='#FFA62F',        # 배경색 설정
    prop={'size':5}           # 텍스트 크기 설정 (이것으로 범례 박스의 크기 간접 조정)
)

# 마진 조정 (왼쪽, 오른쪽, 아래, 위)
plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.9)

# 그래프 전체 배경 색상 설정
plt.gcf().set_facecolor('#FFE8C8')

# 그래프 안쪽 배경 색상 설정
plt.gca().set_facecolor('#FFE8C8')  # 안쪽 배경 색상 (예: 흰색)
plt.show()
plt.clf()


# kaggle submission
test_x = np.array(house_test["GrLivArea"]).reshape(-1,1)
test_x

pred_y = model.predict(test_x) #test셋에 대한 집값
pred_y

#SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y*1000
sub_df

#csv로 바꿔치기
sub_df.to_csv("data/houseprice/sample_submission9.csv", index = False)

house_train['OverallCond'].isna().sum()  
house_test['GarageCars'].isna().sum()

# ====================
# =====   옵션   =====
# ====================

import numpy as np
from scipy.optimize import minimize

# '최소값'을 찾을 다변수 함수 정의
# y = x^2 + 3 의 최소값이 나오는 입력값 구하기 
# # minimize 는 리스트 형식으로 불러드림. 따라서 Y 대신 x 의 인덱싱으로 구성 

# 1
# y = x^2 + 3 
def my_f(x):
    return x ** 2 + 3
my_f(3)

# 초기 추정값
# 함수의 최소값을 찾기 위해 계산을 시작할 초기 점을 설정
initial_guess = [0]

# 최소값 찾기
result = minimize(my_f, initial_guess)  

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

----------------------

# 2
# z = x^2 +y^2 +3
# 2차원 공간에서 함수를 최소화

def my_f2(x):
    return x[0]**2 + x[1]**2 + 3
my_f2([1, 3])

# 초기 추정값
initial_guess = [-10,3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)  

# 결과 출력

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

-----------------------

# 3
# z = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7 
def my_f3(x):
    return (x[0]-1) **2 + (x[1]-2) **2 + (x[2]-4) **2 + 7
my_f3([1, 2, 3])

# 초기 추정값 
initial_guess = [-10, 3, 4]

# 최소값 찾기
result = minimize(my_f3, initial_guess)  

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


------------------------

# 회귀직선 구하기

import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat))) 

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
