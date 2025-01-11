import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


df=pd.read_csv('Hyderabad.csv') #открыть датасет
y=df['Price'].to_list() # цели
X=np.array(df.copy().drop(columns=['Location','Price'],axis=1)) # признаки
reg=LinearRegression().fit(X,y) # обучение модели
x1=np.array([[1000,3]+[0]*(X.shape[1]-2)]) #новый образец
x1_pred=reg.predict(x1) #прогноз
X=np.array([X[i].reshape(1,X.shape[1]) for i in range(X.shape[0])])
yPred=[reg.predict(np.array(X[i])) for i in range(X.shape[0])]
print(mean_squared_error(y, y_pred=yPred)) #среднеквадратическая ошибка
print(mean_absolute_error(y, y_pred=yPred)) #среднеабсолютная ошибка
print(r2_score(y, y_pred=yPred)) #коэффициент детерминации (чем ближе к 1, тем лучше)