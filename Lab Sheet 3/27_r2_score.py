
# Q27: R-squared Score
exec(open("13_multiple_linear_regression.py").read())
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
