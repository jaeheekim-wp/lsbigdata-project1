import pandas as pd
import numpy as np

house_train=pd.read_csv("./data/houseprice/train.csv")
house_train=house_train[["Id", "YearBuilt", "SalePrice"]]
house_train.info()

# 연도별 평균
house_mean=house_train.groupby("YearBuilt", as_index=False) \
                      .agg(mean_year = ("SalePrice", "mean"))
house_mean  

# 테스트 불러오기 
house_test=pd.read_csv("./data/houseprice/test.csv")
house_test=house_test[["Id", "YearBuilt"]]
house_test

house_test=pd.merge(house_test, house_mean, 
                    how="left", on="YearBuilt")
house_test                   
house_test=house_test.rename(
    columns={"mean_year": "SalePrice"}
    )
house_test

# nan 값 갯수 
house_test["SalePrice"].isna().sum()

# 비어있는 테스트 세트 집 목록 확인
house_test.loc[house_test["SalePrice"].isna()]

# 집값 채우기
house_mean=house_train["SalePrice"].mean()
house_test["SalePrice"]=house_test["SalePrice"].fillna(house_mean)

# sub 데이터 불러오기
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# SalePrice 바꿔치기
sub_df["SalePrice"] = house_test["SalePrice"]
sub_df

sub_df.to_csv("./data/houseprice/sample_submission2.csv", index=False)

--------------------------------------------------------------------------------


house_train = pd.read_csv("./data/houseprice/train.csv")
house_train = house_train[["YearBuilt", "OverallCond", "GrLivArea"]]
house_train

# 
house_mean = house_train.groupby("YearBuilt", as_index = False)\
                        .agg(mean_overall = ("OverallCond", "mean"))
house_mean

# merge 
house_test = pd.merge(house_test, house_mean,
                     how='left', on = "YearBuilt")
                     
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

sub_df["SalePrice"] = house_test["SalePrice"]
sub_df


house_test = pd.read_csv("./data/houseprice/test.csv")
house_test = house_test[["Id", "YearBuilt"]]

-# -------------------------------------------------------------------------------

# 재희

train = pd.read_csv("data/houseprice/train.csv")
train = train[["Id","KitchenQual","SalePrice"]]

Kitchen = train.dropna(subset = "KitchenQual") \
                     .groupby("KitchenQual", as_index = False) \
                     .agg(kichen_qual_price = ("SalePrice", "mean"))
  
import seaborn as sns
import matplotlib.pyplot as plt
                     
sns.lineplot(data = Kitchen, x = "KitchenQual", y = "kichen_qual_price") 
plt.show()
plt.clf()

sub_df["SalePrice"] = Kitchen["kichen_qual_price"]
sub_df

# sub 데이터 불러오기
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

# SalePrice 바꿔치기
sub_df["SalePrice"] = Kitchen["kichen_qual_price"]
sub_df

sub_df.to_csv("./data/houseprice/sample_submission3.csv", index=False)

--------------------------------------------------------------------------------

# 현주

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("data/houseprice/train.csv")
train = train[["Id","GarageCars","SalePrice"]]

train["GarageCars"].min()
train["GarageCars"].max()

GC_Price = train.dropna(subset = "GarageCars") \
                     .groupby("GarageCars", as_index = False) \
                     .agg(Price_GC = ("SalePrice", "mean"))
                     
sns.barplot(data = GC_Price, x = "GarageCars", y = "Price_GC") 
plt.show()
plt.clf()

--------------------------------------------------------------------------------

# 수빈

import pandas as pd
import numpy as np


house_df = pd.read_csv("data/houseprice/train.csv")
house_df

#전체적인 품질별 가격 평균 묶기
my_df=house_df.groupby(['OverallQual', 'YearRemodAdd'], as_index=False)\ 
         .agg(SalePrice=('SalePrice', 'mean'))
         
my_df

#테스트 데이터 셋 가져오기         
test_df = pd.read_csv("data/houseprice/test.csv")         

# 테스트 데이터에 평균 집 값을 매핑

test_df=test_df[['Id', 'OverallQual', 'YearRemodAdd']]
test_df
test_df = pd.merge(test_df, my_df, on=['OverallQual', 'YearRemodAdd'], how='left')
test_df

# 결측치 있나 확인
test_df['SalePrice'].isna().sum()

# 결측치 평균으로 처리
mean = house_df["SalePrice"].mean()
test_df["SalePrice"] = test_df["SalePrice"].fillna(mean)

test_df['SalePrice'].isna().sum()

# 필요한 열만 꺼내서 내보내기
test_df = test_df[['Id', 'SalePrice']]
test_df.to_csv("data/houseprice/sample_submission.csv", index=False)


# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(data=test_df, x=test_df['YearRemodAdd'], y=test_df['SalePrice'])
plt.show()
plt.clf()


--------------------------------------------------------------------------------

# 용규

import pandas as pd
import numpy as np
import seaborn as sns    
import matplotlib.pyplot as plt    
import math
import scipy.stats    

house_train_raw = pd.read_csv("data/houseprice/train.csv")
house_train = house_train_raw.copy()
house_train = house_train[["Id","OverallQual","GrLivArea","TotalBsmtSF","SalePrice"]]

# 주택 전반 품질 / 지상 면적 구간 + 지하 면적 구간 > 주택 크기와 가치 평가에 중요 요소 

# 지상 면적 구간 최소/최댓값 
np.min(house_train["GrLivArea"])
np.max(house_train["GrLivArea"])

# 지상 면적 급으로 나눠서 간격 별로 나눔 
cut_lim = [i*200 for i in range(1,29)]
labels = [f"{i*2}-{i*2+2}" for i in range(2, 29)]
num = range(21)
house_train["gr_area"] = pd.cut(house_train["GrLivArea"],bins=cut_lim,labels=labels)
house_train["gr_area"] = house_train["gr_area"].astype(object)

# 총 면적을 급간별로 나눔
# 지상 활동이 많기 때문에 
# 지상은 1 + 지하는 /6 으로 임의 계산 
house_train["tot_area"] = house_train["GrLivArea"]+house_train["TotalBsmtSF"]/6
np.max(house_train["tot_area"])
np.min(house_train["tot_area"])
cut_lim2 = [i*200 for i in range(2,39)]
labels2 = [f"{i*2}-{i*2+2}" for i in range(2, 38)]
house_train["total_area"] = pd.cut(house_train["tot_area"],bins=cut_lim2,labels=labels2)
house_train["total_area"] = house_train["total_area"].astype(object)

'''
df = house_train.groupby(["OverallQual","gr_area"],as_index=False).agg(price_mean = ("SalePrice","mean"))
df2 = house_train.groupby("gr_area",as_index=False).agg(price_mean = (("SalePrice","mean"))).sort_values("price_mean")
'''

# 총면적 급간별로 SALE PRICE 평균값 분리 
# 오름 차순 정렬 
df3 = house_train.groupby("total_area",as_index=False).agg(price_mean = (("SalePrice","mean"))).sort_values("price_mean")
df3['area_start'] = df3['total_area'].apply(lambda x: int(x.split('-')[0]))
df3_after = df3.sort_values('area_start').drop(columns='area_start').reset_index(drop=True)

# 그래프화 
sns.lineplot(data=df3_after,x=df3["total_area"],y=df3["price_mean"])
plt.title("Price by Area")
plt.xlabel("Total Area")
plt.ylabel("Home Price")
plt.show()
plt.clf()
