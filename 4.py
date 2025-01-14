import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score
from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True) #подгружаем датасет ирисов
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y) #логистическая регрессия, 1000 итераций (по дефолту 100)
print(clf.score(X, y)) #средняя точность