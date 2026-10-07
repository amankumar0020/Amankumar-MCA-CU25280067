
# Q25: Mean Squared Error
exec(open("13_multiple_linear_regression.py").read())
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
