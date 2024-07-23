import numpy as np

# 두 개의 벡터를 합쳐 행렬 생성 -세로 
matrix = np.column_stack((np.arange(1, 5),
 np.arange(12, 16)))
print("행렬:\n", matrix)

# 두 개의 벡터를 합쳐 행렬 생성 -가로 
matrix = np.vstack((np.arange(1, 5),
 np.arange(12, 16)))
print("행렬:\n", matrix)


type(matrix)
matrix.shape


#np.zeros((행, 열))
np.zeros(5)
np.zeros([5,4])

#np.reshape((행, 열)-배열의 형태를 지정된 행과 열로 변환
np.arange(1,5).reshape([2,2])
np.arange(1,7).reshape([2,3])
np.arange(1,7).reshape([2,-1])
#-1통해서 크기 자동결정

#0에서 99 중 랜덤 50개 숫자 뽑고 5 by 10 행렬 만들기

np.random.seed(2024)
a=np.random.randint(0,100,50).reshape([5,10])
a

#order 옵션 c- 행우선 f- 열 우선 (세로 ) 
np.arange(1,21).reshape([4,5],order="F") #세로 먼저 채우기 
mat_a = np.arange(1,21).reshape([4,5],order="C") #가로 먼저 채우기 

mat_a[0,0]
mat_a[1,1]
mat_a[2,3]
mat_a[0:2,3] # 행은 두개 0,1 열은 3 
mat_a[1:3,1:4]

 #행/열자리 비어있는 경우 전체 행, 또는 열 선택 
mat_a[3,]
mat_a[3,:]
mat_a[3,::2]


#짝수행만 가져오고 싶을때
mat_b = np.arange(1,101).reshape((20,-1))
mat_b[1::2,:]

mat_b[[1,4,6,14],]

x = np.arange(1, 11).reshape((5, 2)) * 2
print("행렬 x:\n", x)

X[[True]]


mat_b[:,1] #1차원 벡터
mat_b[:,1].reshape((-1,1)) #2차원 행렬
mat_b[:,(1)] #2차원 행렬
mat_b[:,[1]] #2차원 행렬
mat_b[:,1:2] #2차원 유지 


#필터링
mat_b[mat_b[:,1] % 7 == 0,:]
mat_b[mat_b[:,1] > 50,:]

mat_b


import numpy as np
import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


#행렬다운로드 
a = np.random.randint(0,256,20).reshape(4,-1)
a / 255


plt.imshow(a/255, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)
# 행렬을 전치
transposed_x = x.transpose()
print("전치된 행렬 x:\n", transposed_x)

import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

!pip install imageio
import imageio
import numpy as np



# 이미지 읽기
jelly = imageio.imread("img/jelly.png")
jelly
len(jelly)
jelly.shape
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

jelly[:,:,0].shape
jelly[:,:,0].transpose().shape
jelly[:,:,1]
jelly[:,:,2]
jelly[:,:,3]

print("최대값:", jelly.max())
print("최소값:", jelly.min())

plt.imshow(jelly)
plt.imshow(jelly[:,:,0].transpose())
plt.imshow(jelly[:,:,0]) #R
plt.imshow(jelly[:,:,1]) #G
plt.imshow(jelly[:,:,2]) #B
plt.imshow(jelly[:,:,3]) #OPACITY
plt.axis('off') # 축 정보 없애기 
plt.show()
plt.clf()
# RGB
# 첫 3개의 채널은 해당 위치의 빨강, 녹색, 파랑의 색깔 강도를 숫자로 표현
# 마지막 채널은 투명도 opacity


##배열 다루기 
# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)
mat1
mat2

# 3차원 배열로 합치기 >> 겹쳐놓는 것 
my_array = np.array([mat1, mat2])
my_array
my_array.shape  #(2,2,3) 2장 2행 3열 

first_slice = my_array[0, :, :]
print("첫 번째 2차원 배열:\n", first_slice)

my_array[0,1,1:3]


mat_x= np.arange(1,101).reshape((5,5,4))
mat_y= np.arange(1,101).reshape((10,5,2))
mat_y= np.arange(1,101).reshape((-1,5,2))
mat_y= np.arange(1,100).reshape((-1,3,3))

filtered_array = my_array[:, :, :-1]
print("세 번째 요소를 제외한 배열:\n", filtered_array)

my_array2 = np.array([my_array,my_array])
my_array2
my_array2.shape



a= np.array([[1,2,3,],[4,5,6]])
a
a.sum()
a.sum(axis=0) #열 더하기 
a.sum(axis=1) #행 더하기 

a.mean()
a.mean(axis=0) 
a.mean(axis=1)

mat_b = np.random.randint(0,100,50).reshape((5,-1))
mat_b

#가장 큰수
mat_b.max()
mat_b.max(axis=1)
mat_b.max(axis=0)

#누적해서 더하기 
a = np.array([1,3,2,5])  
a.cumprod()

mat_b.cumsum(axis =1 )
mat_b.cumprod(axis =1 )

mat_b.reshape((2,5,5)).flatten()
mat_b.flatten()
type(mat_b)

d = np.array([1, 2, 3, 4, 5])
print("클립된 배열:", d.clip(2, 4))


d_list = d.tolist()
type(d)
type(d_list)


-----------------------
#확률변수: "X" 대문자
#관측값:"x"  소문자 






















