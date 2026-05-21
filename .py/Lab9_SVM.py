import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
data = pd.read_csv('diabetes_data.csv')
#X = data.iloc[:,[0,1,2,3,4,5,6,7]].values
X= data.iloc[0:767].values
Y= data.iloc[0:767,[8]].values
#Y= data.iloc[:,-1].values
print(X)
print(Y)
Y = np.ndarray.flatten(Y)
Y[Y==0]=-1
print(Y) # picks y array and replace it with minus one
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=45)
clf = svm.SVC()#kernel='poly',degree=5) #poly,degree = 2,rbf
clf.fit(X_train,Y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
acc1 = accuracy_score(Y_train,pred_train)
acc2 = accuracy_score(Y_test,pred_test)
print("Training accuracy =",acc1)
print("Testing accuracy =",acc2)