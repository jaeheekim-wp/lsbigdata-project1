import pandas as pd
import numpy as np

house_df=pd.read_csv("./data/houseprice/train.csv")
house_df.shape

price_mean=house_df["SalePrice"].mean() 
price_mean

sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

sub_df["SalePrice"] = price_mean
sub_df

sub_df.to_csv("./data/houseprice/sample_submission.csv", index=False)
sub_df


train_df = pd.read_csv("./data/houseprice/train.csv")
price_mean = house_df.groupby("YearBuilt",as_index = False)\
        .agg(price_mean = ("SalePrice","mean"))
type(df2)
type(df3)

test = pd.read_csv("./data/houseprice/test.csv")
df4 = pd.merge(test,price_mean, how = 'left', on = "YearBuilt")
pd.isna(df4).sum()
df4["price_mean"] =  df4["price_mean"].fillna(methode = ffill)



