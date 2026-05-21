import numpy as np
import pandas as pd
from scipy.ndimage import label
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


from Lab5_KNN import X_test, X_train, Y_test

data=pd.read_csv('weather.csv')
print(data)
le=LabelEncoder()
label=le.fit_transform(data['Outlook'])
data.drop('Outlook',axis=1,inplace=True)
# axis=1 mean column and axis=0 mean rows
data['Outlook']=label
#
#
# label=le.fit_transform(data['Temp'])
# data.drop('Temp',axis=1,inplace=True)
# # axis=1 mean column and axis=0 mean rows
# data['Temp']=label
#
#
#
# label=le.fit_transform(data['Humanity'])
# data.drop('Humanity',axis=1,inplace=True)
# # axis=1 mean column and axis=0 mean rows
# data['Humanity']=label
#
#
# label=le.fit_transform(data['Windy'])
# data.drop('Windy',axis=1,inplace=True)
# # axis=1 mean column and axis=0 mean rows
# data['Windy']=label
#
#
#
# label=le.fit_transform(data['play'])
# data.drop('play',axis=1,inplace=True)
# # axis=1 mean column and axis=0 mean rows
# data['play']=label
# print(data)
#............................................................
print(data)
X=data.iloc[:,[0,1,2,3]].values
Y=data.iloc[:,[4]].values
Y=np.ndarray.flatten(Y)
model=GaussianNB()
cv=LeaveOneOut()
Y_prediction=[]
Y_actual=[]
for train_index,test_index in cv.split(X):
    X_train=X[train_index]
    Y_train=Y[train_index]
    X_test=X[test_index]
    Y_test=Y[test_index]
    model.fit(X_train,Y_train)
    Y_prediction.append(model.predict(X_test))
    Y_actual.append(Y_test[0])
print("Accuracy  of model=",accuracy_score(Y_actual,Y_prediction))
print("precision  of model=",accuracy_score(Y_actual,Y_prediction))
print("Recall  of model=",accuracy_score(Y_actual,Y_prediction))
print(data)