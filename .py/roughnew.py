import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("voice.csv")

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
Y= np.ndarray.flatten(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
clf = tree.DecisionTreeClassifier(criterion='entropy') # Gini
clf.fit(X_train,Y_train)
p = clf.predict(X_test)
p1 = clf.predict(X_train)
print("Accuracy of train",accuracy_score(Y_train,p1))
print("Accuracy of test",accuracy_score(Y_test,p))
plt.figure(figsize=(30,10))
t = tree.plot_tree(clf,fontsize=8)
plt.show()