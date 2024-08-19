# 교재 301 

# 지도 시각화 
import json
geo = json.load(open('data/SIG.geojson', encoding = 'UTF-8'))

# 행정 구역 코드 출력
geo["features"][0]['properties']
# geo["features"][0]['geometry']

# =======================

geo_seoul = json.load(open('data/SIG_Seoul.geojson', encoding = 'UTF-8'))

# 데이터 탐색
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"]
len(["features"])
len(geo_seoul["features"]) 
len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

# 숫자 바뀌면 "구"가 바뀐다 
geo_seoul["features"][0]["type"] # SIG : 
geo_seoul["features"][0]["properties"] 
geo_seoul["features"][0]["geometry"] # 위도/경도 

coordinate_list = geo_seoul["features"][0]["geometry"]["coordinates"] 
len(coordinate_list)
# coordinates 의 value가 4차원 리스트 
len(coordinate_list[0][0])
(coordinate_list[0][0])

# 재정리
len(coordinate_list) # 1 , 대괄호 4개
len(coordinate_list[0])  # 1, 대괄호 3개
len(coordinate_list[0][0]) # 2332개

# 시각화
import numpy as np
coordinate_array=np.array(coordinate_list[0][0]) #x,y 뽑기위해? - 인덱싱하려면 넘파이 구조 ? 다시 물어보기 
coordinate_array
x = coordinate_array[:,0] # 위도 
y = coordinate_array[:,1] # 경도 
len(x)
len(y)

import matplotlib.pyplot as plt

# 종로구 외곽지역 표시 
plt.scatter(x, y)
plt.scatter(x[::3], y[::3]) # 점 갯수 줄이기 
plt.show()
plt.clf()

# 함수 만들기
def draw_seoul(num): 
    gu_name = geo_seoul["features"][num]["properties"]['SIG_KOR_NM']
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"] # 4차원
    coordinate_array=np.array(coordinate_list[0][0]) # 2차원으로 
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    
    plt.rcParams.update({"font.family" : "Malgun Gothic"}) # 출력되기전에 한글 설정 해야함 
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf() 
    
    return None

draw_seoul(1)

# 서울시 전체 지도 그리기 

# 구이름 만들기
# 방법 1 
gu_name=list()
for i in range(25):
    gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
gu_name

# 방법 2 - 리스트 컴프리헨션
gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(len(geo_seoul["features"])))]
gu_name

# x, y 판다스 데이터 프레임 
import pandas as pd

def make_seouldf(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"] # 이름 꺼내기
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"] # 위경도
    coordinate_array=np.array(coordinate_list[0][0]) # 차원 변경 ( 왜? )
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    return pd.DataFrame({"gu_name":gu_name, "x": x, "y": y})

make_seouldf(2)

# 전체 구 합치기 
# 데이터프레임에 넣기

result = pd.DataFrame({})

for i in range(25):
    result = pd.concat([result, make_seouldf(i)], ignore_index = True)

result

# 서울 그래프 그리기
sns.scatterplot(data = result,
                x = 'x', y = 'y',
                hue = 'gu_name', s = 5,
                legend = False,
                palette = 'coolwarm')
plt.show()
plt.clf()

# 팔레트 - 딥/ 색맹용 코드 다시 다운 

# result.plot(kind = "scatter",x="x", y = "y") # y를 선으로 잇는다

# 강남만 다른 색으로 표현
gangnam_df = result.assign(is_gangnam = np.where(result["gu_name"]=="강남구", "강남", "안강남"))
sns.scatterplot(data= gangnam_df, x="x", y = "y", hue = "is_gangnam", s=2,
                legend=False,  palette=["gray", "red"])
plt.show()
plt.clf()

# legend=False : 범례

# ===================

geo_seoul = json.load(open('data/SIG_Seoul.geojson', encoding = 'UTF-8'))
geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("./data/Population_SIG.csv")
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"] = df_seoulpop["code"].astype(str) # 문자형으로 바꿔서 
df_seoulpop.info()


# !pip install folium
import folium

center_x = result["x"].mean()
center_y = result["y"].mean()

# 304p
# 흰 도화지 맵 가져오기 
map_sig = folium.Map(location = [37.55, 126.97],
                     zoom_start = 12,
                     tiles = 'cartodbpositron')

# 계급구간 정하기
bins = list(df_seoulpop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))

# 디자인 수정하기
folium.Choropleth(
    geo_data = geo_seoul, #지도 데이터
    data = df_seoulpop, #통계 데이터 
    columns = ('code', 'pop'), # 행정구역 코드, 인구
    bins = bins,  # 계급구간 기준값
    fill_color = 'viridis', # 컬러맵
    fill_opacity = 0.5, # 투명도
    line_opacity = 0.5, # 경계선 투명도
    key_on = 'feature.properties.SIG_CD') \
    .add_to(map_sig) 
    
    # add_to(map_sig) 지도에 그려라 

# 점 찍는 법 
make_seouldf(0).iloc[:,1:3].mean() # 종로구의 위도 경도 평균 내기 
folium.Marker([37.583744, 126.983800], popup = '종로구').add_to(map_sig)


# 지도 최종 저장 
map_sig.save('map_seoul.html')

# =======================================

# houseprice 
# Longitude(위도)/Latitude(경도)
# 위도 경도 찍기
house_lonlat = pd.read_csv("./data/houseprice/houseprice-with-lonlat.csv")
lonlat = house_lonlat[["Longitude","Latitude"]]
lonlat

# 위/경도 평균 
lon_mean = lonlat["Longitude"].mean()
lat_mean = lonlat["Latitude"].mean()

# 흰 도화지 맵 가져오기 
map_sig = folium.Map(location = [42.034, -93.642],
                     zoom_start = 12,
                     tiles = 'cartodbpositron')
                     
# 지도 내보내기                    
map_sig.save('map_houseprice.html')

# 죄표 찍기 (for 반복문 활용)
for lon, lat in zip(lonlat["Longitude"], lonlat["Latitude"]):
    folium.Marker(location=[lat, lon]).add_to(map_sig)

for idx, row in lonlat.iterrows():
    folium.Marker(location=[row["Latitude"], row["Longitude"]]).add_to(map_sig)

for i in range (len(lonlat)) :
    folium.CircleMarker(location=[lonlat.iloc[i,1], lonlat.iloc[i,0]]).add_to(map_sig)
    
map_sig.save('map_houseprice.html') 


# 집 전부 위/경도 표시하기 
# folium.Marker([42.034, -93.642]).add_to(map_sig) - 하나만 찍을때 
# map_sig.save("map_houseprice.html")


# # 데이터프레임 concat 예제
# df_a = pd.DataFrame({
#     'ID': [],
#     'Name': [],
#     'Age': []
# })
# 
# df_b = pd.DataFrame({
#     'ID': [4, 5, 6],
#     'Name': ['David', 'Eva', 'Frank'],
#     'Age': [40, 45, 50]
# })
# df_a=pd.concat([df_a, df_b])
--------------------------------------------------

# 용규오빠코드
def df_gu(x):
    import numpy as np
    import pandas as pd
    coordinate_list = geo_seoul["features"][x]["geometry"]["coordinates"][0][0]
    coordinate_array = np.array(coordinate_list)
    df = pd.DataFrame({})
    df["gu_name"] = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"]]*len(coordinate_array)
    df["x"] = coordinate_array[:,0]
    df["y"] = coordinate_array[:,1]
    return df
df_gu(0)
result = pd.DataFrame({})
for x in range(len(geo_seoul["features"])):
    result = pd.concat([result,df_gu(x)])
    df = df_gu(x)
    plt.plot(df["x"],df["y"])
plt.show()
result = result.reset_index(drop=True)
result

# 1 
plt.plot(result['x'],result['y'])
#sns.lineplot(data = result,x = 'x', y = 'y', hue= "gu_name")
plt.legend(fontsize = 2)
plt.show()
plt.clf()


# 2
for x in range(len(geo_seoul["features"])):
    result = pd.concat([result,df_gu(x)])
sns.scatterplot(data = result,x = 'x', y = 'y', hue= "gu_name")
plt.legend(fontsize = 2)
plt.show()
plt.clf()


# 승학오빠 코드 
def df_sh(i):
    coordinate_list = geo_seoul["features"][i]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"]]
    df = pd.DataFrame({"x" : coordinate_array[:,0],
                       "y" : coordinate_array[:,1]})
    df["gu_name"] = gu_name * len(coordinate_array)
    df = df[["gu_name","x","y"]]
    return df

df_sh(1)

result =pd.DataFrame({})
for x in range(len(geo_seoul["features"])):
    result = pd.concat([result,df_sh(x)])
result = result.reset_index(drop=True)

# 현욱오빠 코드 
# 빈 리스트 생성
data = []

# 각 구의 이름과 좌표를 리스트에 저장
for i in np.arange(0, 26):
    gu_name = geo_seoul["features"][i]["properties"]['SIG_KOR_NM']
    coordinate_list = geo_seoul["features"][i]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:, 0]
    y = coordinate_array[:, 1]
    
# 각 좌표를 data에 개별 행으로 추가
    for j in range(len(x)):
        data.append({'guname': gu_name,
                     'x': x[j],
                     'y': y[j]})

# 데이터프레임 생성
df = pd.DataFrame(data)

# 데이터프레임 출력
print(df)

