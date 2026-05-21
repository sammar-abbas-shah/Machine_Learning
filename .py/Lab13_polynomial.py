import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import polynomial
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
X = 6 * np.random.rand(200,1) - 3
Y = 0.8 * X**2 + 0.9*X +2 +np.random.randn(200,1)
plt.plot(X,Y,'b.')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
#######################################################
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=40)
lr = LinearRegression()
lr.fit(X_train,Y_train)
y_pred = lr.predict(X_test)
print("R2 score for Linear Regression",r2_score(Y_test,y_pred))
##################################################uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu
plt.plot(X_train,Y_train,'b.',label = 'Training')
plt.plot(X_test,Y_test,'g.',label = 'Testing')
plt.plot(X_train,lr.predict(X_train),color = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
########################################################################################
poly=PolynomialFeatures(degree=2,include_bias=True)###b0
X_train_trans=poly.fit_transform(X_train)
X_test_trans=poly.fit_transform(X_test)
plr= LinearRegression()
plr.fit(X_train_trans,Y_train)
y_pred = plr.predict(X_test_trans)
##y_pred=ndarray.flatten()
print("R2 score for polynomial Regression",r2_score(Y_test,y_pred))
print("coefficients b1,b2=",plr.coef_)
print("b0=",plr.intercept_)
yp=plr.predict(X_train_trans)
yp=np.ndarray.flatten(yp)
plt.plot(X_train,Y_train,'b.',label = 'Training')
plt.plot(X_test,Y_test,'g.',label = 'Testing')
plt.plot(X_train,yp, 'r.',label='prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

