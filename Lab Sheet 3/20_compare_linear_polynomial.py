
# Q20: Compare Linear and Polynomial Regression
exec(open("07_train_linear_regression.py").read())
linear_pred = model.predict(X_test)
poly2 = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_train, y_train)
poly3 = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(X_train, y_train)
print("Linear R2:", r2_score(y_test, linear_pred))
print("Polynomial Degree 2 R2:", r2_score(y_test, poly2.predict(X_test)))
print("Polynomial Degree 3 R2:", r2_score(y_test, poly3.predict(X_test)))
