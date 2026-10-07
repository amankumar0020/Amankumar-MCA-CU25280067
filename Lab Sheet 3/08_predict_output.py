
# Q8: Predict output values
exec(open("07_train_linear_regression.py").read())
y_pred = model.predict(X_test)
print("First 10 predicted values:")
print(y_pred[:10])
