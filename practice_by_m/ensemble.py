from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# !pip install scikit-learn
import sklearn
print(sklearn.__version__)

# 배깅이랑 랜덤포레스트랑 똑같은데디지젼트리했으면 렌덤 포레스트
# 결정트리 vs 랜덤포레스트
# >> 각 분기 결정할 때, 
# 변수를 모두 고려하면 결정트리
# 변수를 랜덤 선택 후 고려하면 랜덤포레스트
 ## model.fit() 학습시 고려하는 변수가 매번 달라짐 

bagging_model = BaggingClassifier(DecisionTreeClassifier(),
                                  n_estimators=50, 
                                  max_samples=100, 
                                  n_jobs=-1, random_state=42)

# * n_estimator: Bagging에 사용될 모델 개수
# * max_sample: 데이터셋 만들때 뽑을 표본크기

# bagging_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf_model=RandomForestClassifier(n_estimators=50, # 트리 갯수 
                                max_leaf_node=16,
                                n_jobs=-1, random_state=42)

# rf_model.fit(X_train, y_train)



