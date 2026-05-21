import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
data=pd.read_csv("iris_data.csv")
X=data.iloc[:,[0,1,2,3]].values
Y=data.iloc[:,[4]]
X=np.asarray(X)
Y=np.asarray(Y)
Y=np.ndarray.flatten(Y)
kf=KFold(n_splits=5,shuffle=True,random_state=42)
clf=KNeighborsClassifier(n_neighbors=5)
acc_score=[]
for tr_index,ts_index in kf.split(X):
    X_train=X[tr_index]#120 samples
    X_test=X[ts_index]#30 samples
    Y_train =Y[tr_index]#120 samples
    Y_test = Y[ts_index]#30 samples
    clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_test)
    acc=accuracy_score(Y_test,Y_pred)
    acc_score.append(acc)
print("Avg accuracy=",sum(acc_score)/5)
