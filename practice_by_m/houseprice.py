---------------HOUSE PRICE

import pandas as pd
import numpy as np

house_df=pd.read_csv("./data/houseprice/train.csv")
house_df.shape
price_mean = house_df["SalePrice"].mean()
price_mean


sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# SALEPRICE 값을 평균값으로 변경
sub_df["SalePrice"] = price_mean
sub_df
sub_df.to_csv("sample_submission.csv")
# 파일 저장 위치에 그대로 덮어쓰기 - Textbook 92장 
sub_df.to_csv("./data/houseprice/sample_submission.csv", index = False) 
