# lec6 행렬

# 벡터(숫자들의 리스트)들을 사용하여 만들 수 있는 객체.

import numpy as np
import pandas as pd

# 두 개의 벡터를 가로로 합쳐 행렬 생성

matrix = np.vstack(
    (np.arange(1, 5),
    np.arange(12, 16))
    )
matrix
print("행렬:\n", matrix)

# 행렬 생성 시 numpy의 np.zeros()나 np.arange(), np.reshape() 같은 함수 사용.
# 행렬을 만들 때 필요한 정보들
# 즉 행렬을 채울 숫자들, 행의 개수, 열의 개수 등을 입력값으로 받습니다.

np.zeros(5)
np.zeros([5, 4])

# 행렬의 크기는 무조건 사각형
# reshape() 함수를 사용하여 행의 수(nrow)와 열의 수 (ncol) 지정
# reshape(행, 열)

np.arange(1, 7).reshape((2, 3))

# -1 통해서 크기를 자동으로 결정할 수 있음

np.arange(1, 7).reshape((2, -1))

# -------------------------------------------
# 예제제
# Q. 0에서 99까지 수 중 랜덤하게 50개 숫자를 뽑아서
# 5 by 10 행렬 만드세요.
np.random.seed(2024)
a = np.random.randint(0, 100, 50).reshape(5, -1)
a

mat_a = np.arange(1, 21).reshape((4, 5), order="F")
mat_a

# 행렬을 채우는 방법 - order 옵션
# order='C': 행 우선 순서 (row-major order), 기본값, C 언어 스타일. 행 먼저 
# order='F': 열 우선 순서 (column-major order), Fortran 언어 스타일, 열 먼저


# 행렬 인덱싱
 ## 인덱싱(Indexing) :  행렬의 특정 원소에 접근하는 방법
 ## 순서쌍 문법은 [row, col]

mat_a[0, 0]
mat_a[1, 1]
mat_a[2, 3]
mat_a[0:2, 3]
mat_a[1:3, 1:4]

# 행자리, 열자리 비어있는 경우 전체 행, 또는 열 선택
mat_a[3, ]
mat_a[3,:]
mat_a[3,::2] #열 전체 중 2스텝으로 추출

# 짝수 행만 선택하려면?
mat_b = np.arange(1, 101).reshape((20, -1))
mat_b[1::2,:] 
 ##행은 1부터 끝까지 2스텝으로, 열은 전체 

# 리스트로 직접 지정해 선택 추출
mat_b[[1, 4, 6, 14], ]

# 인덱싱/슬라이싱으로 선택 추출 

mat_b[:,1]                  # 벡터
mat_b[:,1].reshape((-1, 1)) # 행렬
mat_b[:,(1,)]               # 행렬
mat_b[:,[1]]                # 행렬
mat_b[:,1:2]                # 행렬


# 1부터 10까지의 수에 2를 곱한 값으로 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2


# 행렬 필터링

## true/false 활용
x = np.arange(1, 11).reshape((5, 2)) * 2
print("행렬 x:\n", x)

filtered_elements = x[[True, True, False, False, True], 0]
print("첫 번째 열의 첫 번째, 두 번째, 다섯 번째 행의 원소:\n", filtered_elements)

# True, False를 원소로 가진 배열을 사용하여 행렬에서 원하는 원소를 필터링.
# 예를 들어, x[[True, True, False, False, True], 0]은 
# 행렬 x의 첫 번째 열에서 첫 번째, 두번째, 다섯 번째 행의 원소를 선택하여 반환합니다.


## 조건문 사용한 필터링
filtered_elements = x[x[:, 1] > 15, 0]
print("두 번째 열의 원소가 15보다 큰 행의 첫 번째 열의 원소:\n", filtered_elements)


# x[x[:, 1] > 15, 0]
 ## 행렬 x의 두 번째 열의 원소가 15보다 큰 행의 첫 번째 열의 원소를 선택 반환.
# 두 번째 열의 원소가 15보다 큰 행(들)의 첫 번째 열의 원소 반환


mat_b = np.arange(1, 101).reshape((20, -1))
mat_b[mat_b[:,1] % 7 == 0,:]
mat_b[mat_b[:,1] > 50,:]

mat_b


# 사진은 행렬이다

import numpy as np
import matplotlib.pyplot as plt

# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

# np.random.rand(3, 3)는 0과 1 사이의 실수 난수를 생성하여 3x3 배열을 만듭니다.
# 결과적으로, img1은 3행 3열의 배열이 되며, 각 요소는 0과 1 사이의 난수입니다.

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

a = np.random.randint(0, 256, 20).reshape(4, -1)
a / 255
plt.imshow(a / 255, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)


import urllib.request

img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

# !pip install imageio
import imageio

# 이미지 읽기
jelly = imageio.imread("img/jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

# 이미지 속성
len(jelly)
jelly.shape
jelly.max()
jelly.min())
jelly[:, :, 0].shape
jelly[:, :, 0].transpose().shape


plt.imshow(jelly)
plt.show()
plt.clf()

# plt.imshow(jelly[:, :, 0].transpose())
# plt.imshow(jelly[:, :, 0]) # R
# plt.imshow(jelly[:, :, 1]) # G
# plt.imshow(jelly[:, :, 2]) # B
# plt.imshow(jelly[:, :, 3]) # 투명도
# plt.axis('off') # 축 정보 없애기


# 행렬 연산

#.transpose()
# 행렬 반대로 뒤집기 

# 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)
# 행렬을 전치
transposed_x = x.transpose()
print("전치된 행렬 x:\n", transposed_x)

#transpose() 메서드를 사용하여 차원을 바꿀 수 있습니다. 
# 행렬의 전치(transpose)를 확장한 개념

# 원래 배열
print("원래 배열 my_array:\n", my_array)
my_array.shape
# 차원 변경
transposed_array = my_array.transpose(0, 2, 1)
print("차원이 변경된 배열:\n", transposed_array)


#.dot()
# 행렬의 곱셈 (dot product)  
 ##행렬의 곱셈은 행렬의 크기가 맞아야 가능.
 ##NumPy에서는 dot() 메서드를 사용


# 2행 3열의 행렬 y 생성
y = np.arange(1, 7).reshape((2, 3))
print("행렬 y:\n", y)
# 행렬 x와 y의 크기 확인
print("행렬 x의 크기:", x.shape)
print("행렬 y의 크기:", y.shape)
# 행렬곱 계산
dot_product = x.dot(y)
print("행렬곱 x * y:\n", dot_product)



# 3차원 배열

# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

my_array = np.array([mat1, mat2])
my_array.shape

first_slice = my_array[0, :, :]
first_slice

filtered_array = my_array[:, :, :-1]
filtered_array

my_array[:, :, [0, 2]]
my_array[:, 0, :]
# my_array[0, 1, [1, 2]]
my_array[0, 1, 1:3]


mat_x = np.arange(1, 101).reshape((5, 5, 4))
mat_y = np.arange(1, 100).reshape((-1, 3, 3)) # -1 :  자동 계산 
len(mat_y)

my_array2 = np.array([my_array, my_array])
my_array2[0, :, :, :]
my_array2.shape

# 넘파이 배열 메서드
# #넘파이 메서드 종류= pdf 06-matrix in python 참고

a = np.array([[1, 2, 3], [4, 5, 6]])

a.sum()
a.sum(axis=0)
a.sum(axis=1)


a.mean()
a.mean(axis=0)
a.mean(axis=1)

mat_b=np.random.randint(0, 100, 50).reshape((5, -1))
mat_b

# 가장 큰 수는?
mat_b.max()

# 행별 가장 큰수는?
mat_b.max(axis=1)

# 열별 가장 큰수는?
mat_b.max(axis=0)

a=np.array([1, 3, 2, 5])
a.cumsum()

a=np.array([1, 3, 2, 5])
a.cumprod()

mat_b.cumsum(axis=1)
mat_b.cumprod(axis=1)

mat_b.reshape((2, 5, 5)).flatten()
mat_b.flatten()


d = np.array([1, 2, 3, 4, 5])
d.clip(2, 4)

d_list=d.tolist()
d
d_list

