import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Read the Excel file (make sure the path is correct)
data = pd.read_excel(r"C:\Users\maria\OneDrive\Desktop\dataset_study.xlsx")

# Define features (X) and target (y)
X = data[['study_hours']]  # double brackets = 2D input
y = data['grade']          # target column

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predictions
predicted_score = model.predict(X)

# Evaluation metrics
mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)

# Show results
print("Mean Absolute ERROR (MAE): ", mae)
print("Mean Squared ERROR (MSE): ", mse)
print("Root Mean Squared ERROR (RMSE): ", rmse)

# Predict for 7 hours studied
new_prediction = model.predict([[7]])
print(f"Predicted Score for 7 hours: {new_prediction[0]:.2f}")

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predicted_score, color='red', label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Grade")
plt.title("Study Hours vs Grade")
plt.legend()
plt.show()
