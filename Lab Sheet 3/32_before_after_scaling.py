import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import joblib

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
before = LinearRegression().fit(X_train, y_train)
pred_before = before.predict(X_test)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
after = LinearRegression().fit(X_train_s, y_train)
pred_after = after.predict(X_test_s)
print("R2 Before Scaling:", r2_score(y_test, pred_before))
print("R2 After Scaling:", r2_score(y_test, pred_after))
