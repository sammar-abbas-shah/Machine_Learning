import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

LB=LabelEncoder()
data = pd.read_csv('IG.csv')
data['age']=LB.fit_transform(data['age'])
data['Salary']=LB.fit_transform(data['Salary'])

X = data.iloc[:,[0,1]].values
Y = data.iloc[:,[2]].values
Y= np.ndarray.flatten(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
clf = tree.DecisionTreeClassifier(criterion='entropy') # Gini
clf.fit(X_train,Y_train)
p = clf.predict(X_test)
p1 = clf.predict(X_train)
print("Accuracy of train",accuracy_score(Y_train,p1))
print("Accuracy of test",accuracy_score(Y_test,p))
#f=['surgeries','Glucose','BP','Skin thickness','insuline','BMI','DPF','Age']
f=['age','Salary']
plt.figure(figsize=(30,10))
t = tree.plot_tree(clf,feature_names=f,fontsize=8)
plt.show()