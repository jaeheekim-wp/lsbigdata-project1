from sklearn.metrics import confusion_matrix
import numpy as np

# 아델리: 'A'
# 친스트랩(아델리 아닌것): 'C'
y_true = np.array(['A', 'A', 'C', 'A', 'C', 'C', 'C'])
y_pred = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])

conf_mat=confusion_matrix(y_true=y_true, 
                          y_pred=y_pred,
                          labels=["A", "C"])

conf_mat

from sklearn.metrics import ConfusionMatrixDisplay

p=ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                         display_labels=("Adelie", "Chinstrap"))
p.plot(cmap="Blues")

# 분류모델 성능평가 지표:
#-Accracy
#-Precision
#-Recall
#-F1-score
#-ROC curve

# Precision과 Recall이 상호보완적인 이유:
# Precision을 높이려면 정확하게 맞춘 것만 골라내야 해요. 
# 그러면 많은 걸 빼먹을 수 있어서 Recall이 낮아질 수 있어요.
# Recall을 높이려면 더 많이 맞추기 위해 모든 걸 시도해볼 수 있지만, 
# 그럼 정확도(Precision)가 낮아질 수 있죠

# 즉, 반비례적 특징, 공동의 목표 >> 상호보완적 관계 