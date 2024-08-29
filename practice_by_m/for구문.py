# 구구단 뱉어내기
# 가로 
for i in range(2,10):
    for j in range(1,10):
        print(f'{i}*{j} = {i*j}', end = "\t")
print() 

# 세로
for i in range(1,10):
    for j in range(2,10):
        print(f'{j}*{i} = {i*j}', end = "\t")
print() 

# 1부터 10까지 정수 누적합 구하기 
x = 0
for i in range(1,11):
    x += i
print(x)  

# 1부터 100까지 홀수 누적합 구하기 
x = 0
for i in range(1,100):
    if i%2 == 1:
        x += i
print(x) 

# f 스트링
 # 사람 이름 리스트
names = ["June", "Alice", "Bob"]

# 반복문으로 개별 인사 출력
for x in names:
    print(f'{x}님, 안녕하세요')

a2=np.array(["1반","2반","3반"])
np.random.seed(42)
a2[b2==1]
b2=b[[1,2,3]]
a = np.arange(11)
b = np.arange(3).repeat(3)
a2 == "1반"

np.random.choice(a,size=10,replace=False)
a[[num for num in range(4)]]
a[[b]]
a[[np.arange(3).repeat(3)]]
a[[np.where(b==1,7,0)]]

[num for num in a]
a[4:]