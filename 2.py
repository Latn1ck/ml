import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))#добавить шума к меткам
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1) #регрессия rbf с коэффициентом регуляризации 100
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)#полиномиальная регрессия со степенью 3 с коэффициентом регуляризации 100

yPredRBF=svr_rbf.fit(X, y).predict(X)
yPredPoly=svr_poly.fit(X, y).predict(X)
print('RBF')
print(f'mse error: {mean_squared_error(y, y_pred=yPredRBF)}')
print(f'mse error: {mean_absolute_error(y, y_pred=yPredRBF)}')
print(f'mse error: {r2_score(y, y_pred=yPredRBF)}')
print('Polynomial')
print(f'mse error: {mean_squared_error(y, y_pred=yPredPoly)}')
print(f'mse error: {mean_absolute_error(y, y_pred=yPredPoly)}')
print(f'mse error: {r2_score(y, y_pred=yPredPoly)}')


lw = 2

svrs = [svr_rbf, svr_poly]
kernel_label = ["RBF", "Polynomial"]
model_color = ["m", "g"]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=50,
        label="{} support vectors".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="other training data",
    )
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.04, "data", ha="center", va="center")
fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()
