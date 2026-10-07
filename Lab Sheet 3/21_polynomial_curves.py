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
X = pd.DataFrame(data.data, columns=data.feature_names)[["MedInc"]]
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly2 = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_train, y_train)
poly3 = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(X_train, y_train)
x_plot = np.linspace(X.min().iloc[0], X.max().iloc[0], 300).reshape(-1, 1)
plt.scatter(X_test, y_test, s=5, label="Actual")
plt.plot(x_plot, poly2.predict(x_plot), label="Degree 2")
plt.plot(x_plot, poly3.predict(x_plot), label="Degree 3")
plt.xlabel("Median Income"); plt.ylabel("House Price")
plt.title("Polynomial Regression Curves"); plt.legend(); plt.show()
