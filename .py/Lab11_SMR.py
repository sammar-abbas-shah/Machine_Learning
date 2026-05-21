import numpy as np
import matplotlib.pyplot as plt

X = np.asarray([2, 4, 6, 8])
Y = np.asarray([3, 7, 5, 10])

x_mean = np.mean(X)
y_mean = np.mean(Y)

b1 = np.sum((X - x_mean) * (Y - y_mean)) / np.sum((X - x_mean) ** 2)
b0 = y_mean - b1 * x_mean

x_sample = 5
y_pred = b0 + b1 * x_sample

print(f"b0  = {b0}")
print(f"b1  = {b1}")
print(f"Predicted y for x=5: {y_pred}")
print(f"Y = {b0} + {b1} X")
print(f"When X = {x_sample} then Y = {y_pred}")
# darwaing a scatter plot
x_line = np.linspace(1, 9, 100)
y_line = b0 + b1 * x_line
x_pred = 5
y_pred = b0 + b1 * x_pred
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(x_line, y_line, color='green', label='Regression line')
plt.scatter(x_pred, y_pred, color='red', label=f'Prediction (x=5, y={y_pred:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()