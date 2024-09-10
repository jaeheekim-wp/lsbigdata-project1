import pandas as pd

admission_data = pd.read_csv("./data/admission.csv")
print(admission_data.shape)

# GPA : 학점
# GRE : 입학 시험

# 합격을 한 사건 : Admit    
# Admit의 확률 오즈는? (P(A) / 1 - P(A))

# P(Admit) = 합격 인원 / 전체 학생

p_hat = admission_data['admit'].mean()
p_hat / (1 - p_hat)

# p(A) : 0.5보다 큰 경우 - 오즈비: 무한대에 가까워짐 
# p(A) : 0.5 -> 오즈비 : 1
# p(A) : 0.5보다 작은 경우 -> 0에 가까워짐
# 오즈비 범위 : 0 ~ 무한대

admission_data['rank'].unique()

grouped_data = admission_data \
           .groupby('rank') \
           .agg(p_admit=('admit', 'mean'))
                
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
print(grouped_data)

# 오즈비가 3이면 P(A)?
# 오즈(Odds)를 사용한 확률 역산 Odds가 주어졌을 때, 
# 위의 관계를 사용하여 역으로 확률을 계산
# 𝑝̂ = 𝑂𝑑𝑑𝑠 / 𝑂𝑑𝑑𝑠 + 1


# admission 데이터 산점도 그리기
# x : gre, y : admit
# admission_data

import seaborn as sns
sns.scatterplot(data = admission_data, x = 'gre', y = 'admit')


#로지스틱 문제 풀이 
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.stats import norm
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 문제 1.

# 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
# df = pd.read_csv('./data/leukemia_remission.txt', sep='\t')
df_rem = pd.read_csv("../data/leukemia_remission.csv")
df = df_rem.copy()
df

# "admit ~ gre + gpa + rank + gender"
model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data=df).fit()

print(model.summary())

#문제 2
#해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량을 사용해서 설명하시오.
from scipy.stats import chi2
p_val = 1 - chi2.cdf(-2 * (-17.186 + 10.797), df = 6)
p_val
# LLR p-value:0.04670
# 검정 통계량 12.78
# 1 - chi2.cdf(12.78,6) -> LLR p-value
# 유의 수준 0.05보다 작다
# 결론 : LLR p-value: 0.0467 < 유의수준 0.05보다 작으니까 통계적으로 유의하다고 할 수 있다.

# 문제 3  
# 변수 통계적으로 유의미한게 2개이다.
# z에 대한 p-value 값이 0.2 보다 작은 변수들
# LI TEMP

# 문제 4 
# 다음 환자에 대한 오즈는 얼마인가요?
# CELL : 65%
# SMEAR : 45%
# INFIL : 55%
# LI : 1.2
# BLAST : 1.1세포/μL
# TEMP : 0.9

log_odds = 64.2581 + 30.8301 * 0.65 + 24.6863 * 0.45 + (-24.9745) * 0.55 + 4.3605 * 1.2 +\
           (-0.0115) * 1.1 + (-100.1734) * 0.9
odds = np.exp(log_odds)
odds

# 문제 5 
# 위 환자의 혈액에서 백혈병 세포가 관측되지 않는 확률은 얼마인가요?
p_h = odds / 1 + odds
#0.076


# 문제 6 
# 1이면 관측 안됨을 의미
# TEMP 변수의 계수는 -100.1734.
# 체온이 1 올라가면 로그 오즈는 100.1734만큼 감소
# 백혈병 상태에 도달할 가능성이 크게 줄어든다


#문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
# CELL 변수의 계수/ 표준 오차
cell_coef = 30.8301
cell_std_err = 52.135
# 99% 신뢰구간- Z 값
z_99 = norm.ppf(0.995,0,1)
# 오즈비 신뢰구간
ci_lower = np.exp(cell_coef - z_99 * cell_std_err)
ci_upper = np.exp(cell_coef + z_99 * cell_std_err)

(ci_lower, ci_upper)

#문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
# 예측 확률
pred_remiss = model.predict(df)
# 50% 이상은 1, 이하는 0
pred_value = (pred_remiss >= 0.5).astype(int)
# pred_value = [1 if x >= 0.5 else 0 for x in pred_remiss] 
actual_value = df['REMISS']  
# 혼동 행렬 계산
conf_mat = confusion_matrix(actual_value, pred_value)
p = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = ('관측가능_0', '관측불가_1'))
plt.rcParams['font.family'] = 'Malgun Gothic'

print(conf_mat)
p.plot(cmap="Blues") 
plt.clf()


#문제 9. 해당 모델의 Accuracy는 얼마인가요?
from sklearn.metrics import accuracy_score

(5+15)/(5+3+4+15)  
(5+15)/(5+3+4+15)

accuracy = accuracy_score(actual_value, pred_value)


#문제 10. 해당 모델의 F1 Score를 구하세요.
from sklearn.metrics import f1_score

f1 = f1_score(actual_value, pred_value)
f1

precision = 5/(5+3)
recall = 5/(5+4)

2*(precision * recall) / (precision + recall)   # 0.5882352941176471