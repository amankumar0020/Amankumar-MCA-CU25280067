
# Q15: Predict using testing dataset
exec(open("13_multiple_linear_regression.py").read())
y_pred = model.predict(X_test)
print(y_pred[:10])
