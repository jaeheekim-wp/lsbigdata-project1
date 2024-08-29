import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


# 데이터 불러오기
house_train = pd.read_csv('../data/train.csv')
house_test = pd.read_csv('../data/test.csv')

# NaN 채우기 (수치형 변수)
quantitative = house_train.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

qualitative = house_train.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)

# 통합 데이터프레임 생성 및 더미 코딩
train_n = len(house_train)
df = pd.concat([house_train, house_test], ignore_index=True)

# 의미없는 변수 버리기 
columns_to_drop = ['Street', 'Alley', 'CentralAir', 'Utilities', 'LandSlope', 'PoolQC', 'PavedDrive']

df = df.drop(columns=columns_to_drop)
unknown_counts = (df == "unknown").sum()
columns_to_drop2 = unknown_counts[unknown_counts > 1000].index.tolist()

df = df.drop(columns=columns_to_drop2)

df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Train/Test 데이터셋 분리
train_df = df_imputed.iloc[:train_n, :]
test_df = df_imputed.iloc[train_n:, :]

# X와 y 변수 분리 (Train 데이터)
X = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_scaled, y, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(148, 150, 0.01)
mean_scores = np.zeros(len(alpha_values))

for k, alpha in enumerate(alpha_values):
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)

# 결과를 DataFrame으로 저장
df2 = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df2['lambda'][np.argmin(df2['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 최적의 alpha 값으로 Lasso 모델 학습
lasso_optimal = Lasso(alpha=optimal_alpha)
lasso_optimal.fit(X_scaled, y)

# Test 데이터셋 준비
X_test = test_df.drop(columns=['SalePrice'])
X_test_scaled = scaler.transform(X_test)

# 예측 수행
predictions = lasso_optimal.predict(X_test_scaled)

# 결과를 DataFrame으로 저장
output = pd.DataFrame({'Id': house_test['Id'], 'SalePrice': predictions})

# CSV 파일로 저장
output.to_csv('../data/submission.csv', index=False)
print("Predictions saved to submission.csv")

# ====================================================
#1조
#점수 :  0.15121
#람다 :  169.84


house_train = pd.read_csv('data/train.csv')
house_test = pd.read_csv('data/test.csv')
house_train.info()
house_train.isnull().sum()

################3 결측값 채우기
## train
## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

## test
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected2 = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected2:
    house_test[col].fillna(house_test[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()

train_n=len(house_train)

cols = house_train.select_dtypes(include='object').columns.tolist()

############## 데이터 합치기
df=pd.concat([house_train, house_test]).reset_index(drop=True)
df

#더미 작업
d=pd.get_dummies(df,columns=cols)

train_d = d.iloc[:1460,:]
test_d = d.iloc[1460:,:]
test_d = test_d.drop(columns='SalePrice')

train_dx = train_d.drop(columns='SalePrice')
train_dy = train_d['SalePrice']

############## 알파값 찾아보기
# 알파 값 설정 처음에는 값간격을 크게하고 범위를 넓혀서 찾은후
# 세세한 값을 찾기 위해서 값간격을 작게하고 범위를 좁혀서 세세한 값을 찾는다
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_dx, train_dy, cv = kf,
                                     n_jobs=-1, scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 200, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

############### 학습해서 제출하기
model= Lasso(alpha=169.84)
model.fit(train_dx, train_dy)
y_pred = model.predict(test_d)

submit = pd.read_csv('data/sample_submission.csv')
submit['SalePrice'] = y_pred

# ===============================================
# 4조
# 점수 = 0.15534
# lambda = 166
# 필요한 데이터 불러오기

house_train = pd.read_csv("./data/train.csv")
house_test = pd.read_csv("./data/test.csv")
sub_df = pd.read_csv("./data/sample_submission.csv")

# NaN 채우기
# 각 숫자변수는 평균 채우기
quantitative = house_train.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

# 각 범주형 변수는 최빈값으로 채우기
qualitative = house_train.select_dtypes(include="object")
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    mode_value = house_train[col].mode()[0]
    house_train[col].fillna(mode_value, inplace=True)

# 동일하게 테스트 데이터에도 NaN 채우기 수행
quantitative_test = house_test.select_dtypes(include=[int, float])
quant_selected_test = quantitative_test.columns[quantitative_test.isna().sum() > 0]

for col in quant_selected_test:
    house_test[col].fillna(house_test[col].mean(), inplace=True)

# 범주형 변수에 대해 NaN을 최빈값으로 채우기
qualitative_test = house_test.select_dtypes(include="object")
qual_selected_test = qualitative_test.columns[qualitative_test.isna().sum() > 0]

for col in qual_selected_test:
    mode_value_test = house_test[col].mode()[0]
    house_test[col].fillna(mode_value_test, inplace=True)

# train과 test 데이터를 결합하여 더미 코딩 수행
combined = pd.concat([house_train, house_test], keys=['train', 'test'])

# 결합된 데이터를 더미 코딩
combined = pd.get_dummies(combined, drop_first=True)

# 다시 train과 test로 분리
house_train = combined.xs('train')
house_test = combined.xs('test')

# house_train에서 SalePrice를 y로, 나머지를 x로 분리
house_train_y = house_train["SalePrice"]  # SalePrice 열만 가져옴
house_train_x = house_train.drop(columns=["SalePrice", "Id"])  # SalePrice와 Id을 제외한 나머지 열들만 가져옴

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, house_train_x, house_train_y, cv=kf,
                                     scoring="neg_mean_squared_error").mean())
    return(score)

lasso = Lasso(alpha=0.01)
rmse(lasso)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 1000, 1)
mean_scores = np.zeros(len(alpha_values))

k = 0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 최적의 alpha 값으로 모델 학습
model = Lasso(alpha=optimal_alpha)
model.fit(house_train_x, house_train_y)

# house_test_x 준비
house_test_x = house_test.drop(columns=["SalePrice", "Id"])

# train과 test에서 차이가 나는 열 확인
extra_columns_in_test = set(house_test_x.columns) - set(house_train_x.columns)
extra_columns_in_train = set(house_train_x.columns) - set(house_test_x.columns)

extra_columns_in_test, extra_columns_in_train

# 테스트 데이터에 대해 예측 수행
pred_y = model.predict(house_test_x)

# 예측 결과를 제출 파일에 반영
sub_df["SalePrice"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("./data/sample_submission777.csv", index=False)