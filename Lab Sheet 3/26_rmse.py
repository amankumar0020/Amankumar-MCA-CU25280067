
# Q26: Root Mean Squared Error
exec(open("13_multiple_linear_regression.py").read())
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
