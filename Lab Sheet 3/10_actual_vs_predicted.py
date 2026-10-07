
# Q10: Compare actual and predicted values
exec(open("07_train_linear_regression.py").read())
y_pred = model.predict(X_test)
comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print(comparison.head(10))
