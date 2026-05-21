import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('taxi-fares.csv')

# Print column names
print("Columns in dataset:", df.columns)

# Define target and features
target = 'fare_amount'
features = df.drop(columns=[target])
X = features.select_dtypes(include=[np.number])
y = df[target]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict
y_pred = model.predict(X_test_poly)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
corr_coef = np.corrcoef(y_test, y_pred)[0, 1]

# Print metrics
print("\nPolynomial Regression Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Correlation Coefficient: {corr_coef:.4f}")

# Plot: Actual vs Predicted with custom color
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color="#FF5733", edgecolor="pink", alpha=0.6, s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="yellow", linestyle="--", linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Fare", fontsize=12)
plt.ylabel("Predicted Fare", fontsize=12)
plt.title("Polynomial Regression: Actual vs Predicted Fare", fontsize=14, color="#333333")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()