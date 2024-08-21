import numpy as np
import matplotlib.pyplot as plt
import json
# !pip install folium
import folium
import pandas as pd
# JSON 파일 형식으로 저장된 지리정보(GeoJSON)

# 지도 시각화 
import json
geo = json.load(open('data/SIG.geojson', encoding = 'UTF-8'))

# 행정 구역 코드 출력
geo["features"][0]['properties']
# geo["features"][0]['geometry']

# ========================

# 서울 지도 그리기 
# 데이터 불러오기
geo_seoul = json.load(open("./data/SIG_Seoul.geojson", encoding="UTF-8"))

# 데이터 탐색
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])  # 25개 구역 
len(geo_seoul["features"][0]) # 각 구가 가지고 있는 key 갯수  
geo_seoul["features"][0].keys() # 각 구가 가지고 있는 key 목록 

# 숫자가 바뀌면 "구"가 바뀌는구나!
geo_seoul["features"][2]["properties"]
geo_seoul["features"][2]["properties"].keys()
geo_seoul["features"][0]["geometry"]
geo_seoul["features"][0]["geometry"].keys()

# 리스트로 정보 빼오기
coordinate_list=geo_seoul["features"][2]["geometry"]["coordinates"]  
type(coordinate_list)
len(coordinate_list[0][0])
coordinate_list[0][0]
# coordinates(좌표) 의 value가 4차원 '리스트' 

# 재정리
len(coordinate_list) # 1 , 대괄호 4개
len(coordinate_list[0])  # 1, 대괄호 3개
len(coordinate_list[0][0]) # 2332개


coordinate_array=np.array(coordinate_list[0][0]) # NumPy 배열로 변환
coordinate_array
x=coordinate_array[:,0]
y=coordinate_array[:,1]

plt.plot(x, y)
plt.show()
plt.clf()

# 종로구 외곽지역 표시 
plt.scatter(x, y)
plt.scatter(x[::3], y[::3]) # 점 갯수 줄이기 
plt.show()
plt.clf()

# 함수로 만들기
def draw_seoul(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    # 축 비율 1:1로 설정
    plt.axis('equal')
    plt.show()
    plt.clf()
    
    return None

draw_seoul(12)


# 서울시 전체 지도 그리기
# gu_name | x | y
# ===============
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# ......
# 종로구  | 126 | 36
# 종로구  | 126 | 36
# 중구  | 126 | 36
# 중구  | 126 | 36
# ......
# 중구  | 126 | 36

# x, y 판다스 데이터 프레임 

# # 구이름 만들기
# # 방법 1 반복문
# gu_name=list()
# for i in range(25):
#     gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
# gu_name
# 
# # 방법 2 리스트 컴프리헨션 
# gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(25))]
# gu_name

import pandas as pd

def make_seouldf(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    return pd.DataFrame({"gu_name":gu_name, "x": x, "y": y})

make_seouldf(1)

result=pd.DataFrame({})
for i in range(25):
    result=pd.concat([result, make_seouldf(i)], ignore_index=True)    

result

# 서울 그래프 그리기
import seaborn as sns
sns.scatterplot(data=result,
    x='x', y='y', hue='gu_name', legend=False,
    palette="viridis", s=2)
plt.show()
plt.clf()


# 서울 그래프 그리기
# 강남만 다른 색으로 표현
import seaborn as sns
gangnam_df=result.assign(is_gangnam=np.where(result["gu_name"]!="강남구", "안강남", "강남"))
sns.scatterplot(
    data=gangnam_df,
    x='x', y='y', legend= False, 
    palette={"안강남": "#F7B787", "강남": "#527853"},
    hue='is_gangnam', s=2)

# 마진 조정 (왼쪽, 오른쪽, 아래, 위)
# plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.9)

# 그래프 전체 배경 색상 설정
plt.gcf().set_facecolor('#F9E8D9')

# 그래프 안쪽 배경 색상 설정
plt.gca().set_facecolor('#FFFFFF')  # 안쪽 배경 색상 (예: 흰색)

# 폰트 크기 조정 (예시)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('x', fontsize=4)
plt.ylabel('y', fontsize=4)
plt.title('gangnam spot', fontsize=7)

# x축과 y축 레이블 회전 (필요 시)
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)
plt.show()
plt.clf()

# legend=False : 범례

# result.plot(kind = "scatter",x="x", y = "y") # y를 선으로 잇는다

# # 데이터프레임 concat 예제
# df_a = pd.DataFrame({
#     'ID': [1, 2, 3],
#     'Name': ['David', 'Eva', 'Frank'],
#     'Age': [35, 45, 55]
# })
# 
# df_b = pd.DataFrame({
#     'ID': [4, 5, 6],
#     'Name': ['David', 'Eva', 'Frank'],
#     'Age': [40, 45, 50]
# })
# df_a=pd.concat([df_a, df_b])

# 시군구별 인구 단계 구분도 
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

geo_seoul = json.load(open("./data/SIG_Seoul.geojson", encoding="UTF-8"))
geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("data/Population_SIG.csv")
df_seoulpop=df_pop.iloc[1:26]
df_seoulpop["code"]=df_seoulpop["code"].astype(str) # 행정구역 코드 문자형이여야 지도 만들수 있음.
df_seoulpop.info()

# 단계 구분도 그리기 
# 패키지 설치 
# !pip install folium
import folium

center_x=result["x"].mean()
center_y=result["y"].mean()

# p.304
# 흰 도화지 맵 가져오기

map_sig=folium.Map(location = [37.551, 126.973], # 지도 중심 좌표
                  zoom_start=12, # 확대 단계 
                  tiles="cartodbpositron") # 지도 종류 

# 코로플릿 사용해서 - 구 경계선 그리기
folium.Choropleth(
    geo_data=geo_seoul,
    data=df_seoulpop,
    columns=("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
    
# map_sig.save('map_seoul.html')

# 계급구간 정하기
bins = list(df_seoulpop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))

# 디자인 수정하기
folium.Choropleth(
    geo_data = geo_seoul, #지도 데이터
    data = df_seoulpop, #통계 데이터 
    columns = ('code', 'pop'), # 색으로 표현할 변수 (행정구역 코드, 인구)
    bins = bins,  # 계급구간 기준값
    fill_color = 'OrRd', # 컬러맵
    fill_opacity = 0.5, # 투명도
    line_opacity = 0.5, # 경계선 투명도
    key_on = 'feature.properties.SIG_CD') \
    .add_to(map_sig) 
    
    # add_to(map_sig) 지도에 그려라 

# 좌표 찍는 법 
make_seouldf(0).iloc[:,1:3].mean() # 종로구의 위도 경도 평균 내기 
folium.Marker([37.583744, 126.983800], popup = '종로구').add_to(map_sig)


# 지도 최종 저장 
map_sig.save('map_seoul2.html')


# 코로플릿 with bins
# matplotlib 팔레트
# tab10, tab20, Set1, Paired, Accent, Dark2, Pastel1, hsv 

# seaborn 팔레트
# deep, muted, bright, pastel, dark, colorblind, viridis, inferno, magma, plasma

# Folium에서 colorbrewer 컬러맵
# 'YlOrRd' (Yellow-Orange-Red)
# 'BuPu' (Blue-Purple)
# 'BuGn' (Blue-Green)
# 'GnBu' (Green-Blue)
# 'OrRd' (Orange-Red)
# 'PuBu' (Purple-Blue)
# 'RdPu' (Red-Purple)
# 'YlGn' (Yellow-Green)
# 'YlGnBu' (Yellow-Green-Blue)
# 'YlOrBr' (Yellow-Orange-Brown)

# ============================================================

# houseprice 

# Longitude(경도)/Latitude(위도)
# 위도 경도 찍기
house_lonlat = pd.read_csv("./data/houseprice/houseprice-with-lonlat.csv")
lonlat = house_lonlat[["Longitude","Latitude"]]
lonlat

# 위/경도 평균 
lat_mean = lonlat["Latitude"].mean()
lon_mean = lonlat["Longitude"].mean()


# 흰 도화지 맵 가져오기 
map_sig = folium.Map(location = [42.034, -93.642],
                     zoom_start = 12,
                     tiles = 'cartodbpositron')
                     
# 지도 내보내기                    
# map_sig.save('map_houseprice.html')

# 집값 전체 좌표 찍기 (for 반복문 활용)
# 마커 클러스터 추가
# marker_cluster = MarkerCluster().add_to(map_sig)

for i in range(len(lonlat)):
    folium.CircleMarker(location=[lonlat.iloc[i, 1], lonlat.iloc[i, 0]],
                        radius=5,
                        color='#B4D6CD',
                        fill=True,
                        fill_color='#B4D6CD').add_to(map_sig)

# 특정 구역 표시 

# circle
#location1=lonlat.iloc[0:5,1].mean()
#location2=lonlat.iloc[0:5,0].mean()
folium.Circle(
    location=[42.054, -93.623],
    radius=500,  # 반경 (미터 단위)
    color='#FFDA76',
    fill=True,
    fill_color='#FFDA76'
).add_to(map_sig)

# Polygon
# location3=lonlat.iloc[2925:-1,1].mean()
# location4=lonlat.iloc[2925:-1,0].mean()
# folium.Polygon(
#     locations=[41.988,-93.603],[41.998, -93.603], [41.988 , -93.613]],
#     color='#FF8C9E',
#     fill=True,
#     fill_color='#FF8C9E'
# ).add_to(map_sig)

# Marker with custom icon
folium.Marker(
    location=[42.054, -93.623],
    popup='expensive',
    icon=folium.Icon(icon='cloud', color='pink')
).add_to(map_sig)

## {'gray', 'beige', 'darkpurple', 'lightgreen', 'darkgreen', 'purple', 
# 'darkred', 'cadetblue', 'darkblue', 'black', 'lightred', 'red', 'orange', 
# 'pink', 'white', 'blue', 'lightblue', 'green', 'lightgray'}.

# LatLngPopup 개별 위경도 팝업 
folium.LatLngPopup().add_to(map_sig)

# 지도 저장
map_sig.save('map_houseprice.html') 


# zip():
# for lon, lat in zip(lonlat["Longitude"], lonlat["Latitude"]):
#     folium.Marker(location=[lat, lon]).add_to(map_sig)

# iterrows():
# for idx, row in lonlat.iterrows():
#     folium.Marker(location=[row["Latitude"], row["Longitude"]]).add_to(map_sig)


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

