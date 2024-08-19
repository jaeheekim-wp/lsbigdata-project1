# 숙제 확인 코드 

# 구글 시트 불러오기 

import pandas as pd
ghseetid = "1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8"
sheet_name = 'Sheet2'
gsheet_url = "https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=sheet2"
df = pd.read_csv(gsheet_url)
df.head()

# EDIT 뒷부분 변경해줘야 함 

# 랜덤 2명 보여주는 코드 

import numpy as np
np.random.seed(20240730)
np.random.choice(df["이름"], 2, replace=False)







