import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



# Load dataset
Data = pd.read_csv('diamonds.csv')
print("Details about data")
print(Data.info())

# Drop rows with problematic values
Data = Data.drop(Data[Data['x'] == 0].index)
Data = Data.drop(Data[Data['y'] == 1].index)
Data = Data.drop(Data[Data['z'] == 2].index)

# Identify object columns for label encoding
s = (Data.dtypes == 'object') | (Data.dtypes == 'string')
object_cols = list(s[s].index)
print("Object columns to encode:", object_cols)

# Label encode categorical features
lb = LabelEncoder()
for c in object_cols:
    Data[c] = lb.fit_transform(Data[c])

# Split features and target
X = Data.drop(['price'], axis=1)
Y = Data['price']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=100)

# # Train Linear Regression model
# lr = LinearRegression()
# lr.fit(X_train, Y_train)
#
# # Predictions
# pred = lr.predict(X_test)
# pred2 =lr.predict(X_train)
#
# # Evaluation
# print("MSE:", mean_squared_error(Y_test, pred))
# print("MAE:", mean_absolute_error(Y_test, pred))
# print("R2 Score:", r2_score(Y_test, pred))
#
# # Plotting
# plt.scatter(Y_train, pred2, color='red', label='Train Data')
# plt.scatter(Y_test, pred, color='green',label='test data')
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Linear Regressor - Price Prediction")
# plt.legend()
# plt.show()
#

from sklearn import linear_model
from sklearn.linear_model import Lasso

reg1=Lasso(alpha=2)
reg1.fit(X_train,Y_train)
pred1=reg1.predict(X_test)
print("Coefficient Of lasso",reg1.coef_)
print("b0 coefficinet",reg1.intercept_)

print("MSE",mean_squared_error(Y_test,pred1))
print("MAE",mean_absolute_error(Y_test,pred1))
print("R2 score",r2_score(Y_test,pred1))

####################################################
reg2=Ridge(alpha=2)
reg2.fit(X_train,Y_train)
pred1=reg1.predict(X_test)
print("Coefficient Of lasso",reg1.coef_)
print("b0 coefficinet",reg1.intercept_)

print("MSE",mean_squared_error(Y_test,pred1))
print("MAE",mean_absolute_error(Y_test,pred1))
print("R2 score",r2_score(Y_test,pred1))

##################################################
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-4,4,0.05)
def sigmoid(z):
    s = 1.0/(1.0 + np.exp(-z))
    return s
plt.plot(z)
output = sigmoid(z)
plt.plot(z,output)
plt.grid()
plt.legend()
plt.show()
