
# Q24: Mean Absolute Error
exec(open("13_multiple_linear_regression.py").read())
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
