import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Illustrative sample data only; prices are in lakh rupees.
data = {
    "Area": [800,1000,1200,1500,1800,2000,900,1100,1400,1600,2200,2500,1300,1700,1900,2100,950,1250,1550,2300],
    "Bedrooms": [2,2,3,3,3,4,2,2,3,3,4,4,3,3,4,4,2,3,3,4],
    "Bathrooms": [1,2,2,2,3,3,1,2,2,2,3,4,2,3,3,3,1,2,2,4],
    "Location": ["Suburb","Suburb","City","City","City","City","Suburb","Suburb","City","City","City","Premium","Suburb","City","Premium","Premium","Suburb","City","City","Premium"],
    "Parking": [0,1,1,1,2,2,0,1,1,1,2,2,1,2,2,2,0,1,1,2],
    "Age": [15,10,8,5,4,3,20,12,7,6,2,1,9,5,3,2,18,8,6,2],
    "Price": [25,32,45,60,75,90,28,35,52,65,100,130,42,70,85,105,27,44,63,115],
}
df = pd.DataFrame(data)
df.to_csv("house_price_dataset.csv", index=False)

print("First five rows:")
print(df.head())
print("\\nMissing values:")
print(df.isnull().sum())

# Basic preprocessing.
df = df.drop_duplicates().copy()
df["Location"] = df["Location"].fillna(df["Location"].mode()[0])
for column in df.select_dtypes(include="number").columns:
    df[column] = df[column].fillna(df[column].median())

# Encode the categorical Location column.
df = pd.get_dummies(df, columns=["Location"], dtype=int)
X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\\n{name}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
    print(f"MSE: {mean_squared_error(y_test, predictions):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.3f}")
    print(f"R2: {r2_score(y_test, predictions):.3f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, models["Linear Regression"].predict(X_test), label="Linear Regression")
plt.scatter(y_test, models["Random Forest"].predict(X_test), label="Random Forest")
plt.xlabel("Actual price (lakh rupees)")
plt.ylabel("Predicted price (lakh rupees)")
plt.title("House Price: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("house_price_prediction_plot.png", dpi=150)
plt.show()

joblib.dump(
    {"model": models["Random Forest"], "features": list(X.columns)},
    "house_price_model.pkl"
)
print("\\nSaved model: house_price_model.pkl")

# Prepare new-house input with the same feature columns/order used in training.
new_house = pd.DataFrame([{
    "Area": 1800,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Parking": 1,
    "Age": 5,
    "Location_City": 1,
    "Location_Premium": 0,
    "Location_Suburb": 0,
}])
new_house = new_house.reindex(columns=X.columns, fill_value=0)
print("Predicted house price (lakh rupees):", round(models["Random Forest"].predict(new_house)[0], 2))
