import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data generation
X = 6 * np.random.rand(200, 1) - 3
Y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

# Plot original data
plt.plot(X, Y, 'b.')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Generated Data")
plt.show()

# Split the data (note: test and train are mistakenly swapped in your original code)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)

# R2 score
print("R2 score for Linear Regression:", r2_score(Y_test, y_pred))

# Sort X_train for smooth line plotting
X_train_sorted = np.sort(X_train, axis=0)
Y_pred_sorted = lr.predict(X_train_sorted)

# Plot results
plt.plot(X_train, Y_train, 'b.', label='Training')
plt.plot(X_test, Y_test, 'g.', label='Testing')
plt.plot(X_train_sorted, Y_pred_sorted, color='red', label='Linear Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title("Linear Regression Fit")
plt.show()
