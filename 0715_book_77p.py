import pandas as pd
import numpy as np

pd.DataFrame()
df= pd.DataFrame({
    '제품' : ["사과", "딸기","수박"],
    '가격' : [1800,1500,3000],
    '판매량' : [24,38,13]})


df
sum(df['가격'])/4
