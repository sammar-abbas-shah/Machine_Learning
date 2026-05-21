import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
data = pd.read_csv("Weather.csv")
print(data)
le = LabelEncoder()
label = le.fit_transform(data['Outlook'])
data.drop('Outlook',axis=1,inplace = True)
data['Outlook']=label
print(data)
label = le.fit_transform(data['Temp'])
data.drop('Temp',axis=1,inplace = True)
data['Temp']=label
print(data)
label = le.fit_transform(data['Humidity'])
data.drop('Humidity',axis=1,inplace = True)
data['Humidity']=label
print(data)
label = le.fit_transform(data['Windy'])
data.drop('Windy',axis=1,inplace = True)
data['Windy']=label
print(data)
label = le.fit_transform(data['Play'])
data.drop('Play',axis=1,inplace = True)
data['Play']=label
print(data)
###################################
print(data)
X=data.iloc[:,[0,1,2,3]].values #only pick values
Y=data.iloc[:,[4]].values
Y=np.ndarray.flatten(Y)
model = GaussianNB()
cv= LeaveOneOut()
Y_prediction=[]
Y_actual=[]
for train_indx,test_indx in cv.split(X):# 14 times
    X_train = X[train_indx]
    Y_train = Y[train_indx]
    X_test = X[test_indx]
    Y_test = Y[test_indx]
    model.fit(X_train,Y_train)
    Y_prediction.append(model.predict(X_test))
    Y_actual.append(Y_test[0])
print("Accruacy of Model = ",accuracy_score(Y_actual,Y_prediction))
