import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Illustrative sample data only; replace with a real dataset for meaningful results.
data = {
    "Attendance": [90,80,75,95,60,85,70,88,92,65,78,82,96,72,89,68,94,77,84,91],
    "Internal_Marks": [18,15,14,19,10,17,12,18,19,11,15,16,20,13,18,12,19,14,17,19],
    "Assignment_Marks": [9,8,7,10,5,9,6,9,10,6,8,8,10,7,9,6,10,7,8,9],
    "Study_Hours": [5,3,2,6,1,4,2,5,6,2,3,4,7,2,5,2,6,3,4,5],
    "Previous_Marks": [85,70,65,90,50,80,60,85,92,55,72,75,95,62,88,58,91,68,78,86],
    "Final_Marks": [88,72,68,94,52,82,63,87,95,57,74,78,97,65,90,60,93,70,80,89],
}
df = pd.DataFrame(data)
df.to_csv("student_performance_dataset.csv", index=False)

print("First five rows:")
print(df.head())
print("\\nMissing values:")
print(df.isnull().sum())

# Basic preprocessing: remove duplicate rows and fill missing numeric values with column medians.
df = df.drop_duplicates().copy()
df = df.fillna(df.median(numeric_only=True))

X = df.drop(columns=["Final_Marks"])
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }
    print(f"\\n{name}")
    for metric, value in results[name].items():
        print(f"{metric}: {value:.3f}")

# Plot actual vs predicted values for both models.
plt.figure(figsize=(8, 5))
plt.scatter(y_test, models["Linear Regression"].predict(X_test), label="Linear Regression")
plt.scatter(y_test, models["Random Forest"].predict(X_test), label="Random Forest")
plt.xlabel("Actual final marks")
plt.ylabel("Predicted final marks")
plt.title("Student Performance: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("student_prediction_plot.png", dpi=150)
plt.show()

# Save the Random Forest model and feature order.
joblib.dump(
    {"model": models["Random Forest"], "features": list(X.columns)},
    "student_model.pkl"
)
print("\\nSaved model: student_model.pkl")

new_student = pd.DataFrame([{
    "Attendance": 85,
    "Internal_Marks": 17,
    "Assignment_Marks": 9,
    "Study_Hours": 4,
    "Previous_Marks": 80,
}])
new_student = new_student[list(X.columns)]
print("Predicted final marks:", round(models["Random Forest"].predict(new_student)[0], 2))
