# !pip install plotly

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

df_covid19_100=pd.read_csv("../data/df_covid19_100.csv")

df_covid19_100.info()

fig = go.Figure(
    data = {"type": "scatter",
         "mode": "markers",
         "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
         "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
         "marker": {"color": "#96C9F4"}
         }
)

#그래프 출력
# fig.show()


# p.26 마진 변수 설정
margins_P = {"l": 30, "r": 30, "t": 70, "b": 70}
fig = go.Figure(
    data = [
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
            "marker": {"color": "#FFB4C2"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"]=="KOR", "new_cases"],
            "line": {"color": "#B1AFFF", "dash": "dash"}
        }
    ],
    layout= {
        "title": "코로나 19 발생현황",
        "xaxis": {"title": "날짜", "showgrid": False},
        "yaxis": {"title": "확진자수"},
        "margin": margins_P
    }
)


# 레이아웃 설정
fig.update_layout(
    plot_bgcolor="#FFF9D0",  # 플롯 배경색 (플롯 영역)
    paper_bgcolor="#CAF4FF" # 전체 배경색 (전체 그래프 영역)
    
)
# x축과 y축의 그리드 라인의 색상
fig.update_xaxes(showgrid=True, gridcolor='#F5DAD2')
fig.update_yaxes(showgrid=True, gridcolor='#F5DAD2')

# 그래프 출력
fig.show()


# 프레임속성을 이용한 애니메이션
# 애니메이션 프레임 생성
frames = []
dates = df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"].unique()
for date in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(date)
    }
    frames.append(frame_data)
    

# x축과 y축의 범위 설정
x_range = ['2022-10-03', '2023-01-11']
y_range = [8900, 88172]


# 애니메이션을 위한 레이아웃 설정
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False, "range": x_range},
    "yaxis": {"title": "확진자수", "range": y_range},
    "margin": margins_P,
    "updatemenus": [{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
        }, {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        }]
    }]
}

# Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout,
    frames=frames
)

fig.show()



