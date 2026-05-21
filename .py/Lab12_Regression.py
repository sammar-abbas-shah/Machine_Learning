import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('diamonds.csv')
print("Details about data")
print(data.info())
data = data[data['x'] != 0]
data = data[data['y'] != 0]
data = data[data['z'] != 0]
s = (data.dtypes == 'object') | (data.dtypes == 'string')
print(s)
object_cols = list(s[s].index)
print(object_cols)
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for c in object_cols:
    data[c] = lb.fit_transform(data[c])
X = data.drop(['price'], axis=1)
Y = data['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=75)
lr = LinearRegression()
lr.fit(X_train, Y_train)
pred_test = lr.predict(X_test)
pred_train = lr.predict(X_train)
print("MSE =", mean_squared_error(Y_test, pred_test))
print("MAE =", mean_absolute_error(Y_test, pred_test))
plt.scatter(Y_train, pred_train, color='red', label='Train Predictions')
plt.scatter(Y_test, pred_test, color='green', label='Test Predictions')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.title("Actual vs Predicted Diamond Prices")
plt.show()