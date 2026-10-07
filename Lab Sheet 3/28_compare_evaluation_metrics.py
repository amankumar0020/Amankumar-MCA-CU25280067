
# Q28: Compare Linear and Polynomial Regression using metrics
exec(open("07_train_linear_regression.py").read())
linear_pred = model.predict(X_test)
poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_train, y_train)
poly_pred = poly.predict(X_test)
for name, pred in [("Linear", linear_pred), ("Polynomial Degree 2", poly_pred)]:
    print("\n", name)
    print("MAE:", mean_absolute_error(y_test, pred))
    print("MSE:", mean_squared_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2:", r2_score(y_test, pred))
