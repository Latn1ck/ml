from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score
import numpy as np


X, y = load_digits(return_X_y=True)
clf = Perceptron(penalty='l1')
clf.fit(X, y)
print(clf.score(X, y))
X=np.array([X[i].reshape(1,X.shape[1]) for i in range(X.shape[0])])
yPred=[clf.predict(np.array(X[i])) for i in range(X.shape[0])]