from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) #разбиение датасета на train и test (поровну)
print(len(X_test)/len(X_train))
gnb = GaussianNB() #гауссовский наивный байесовский алгоритм (предполагаем, что признаки нормально распределены)
y_pred = gnb.fit(X_train, y_train).predict(X_test) #обучение и прогноз
print(f'Number of mislabeled points out of a total {X_test.shape[0]} points : {(y_test != y_pred).sum()}')
#оценки модели
print(f'Precision: {precision_score(y_test, y_pred=y_pred,average='macro')}')
print(f'Recall: {recall_score(y_test, y_pred=y_pred,average='macro')}')
print(f'F1: {f1_score(y_test, y_pred=y_pred,average='macro')}')
print(f'Accuracy: {accuracy_score(y_test, y_pred=y_pred)}')
print(gnb.score(X,y))