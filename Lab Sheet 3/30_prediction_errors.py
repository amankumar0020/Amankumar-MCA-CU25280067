
# Q30: Visualize prediction errors
exec(open("13_multiple_linear_regression.py").read())
y_pred = model.predict(X_test)
errors = y_test - y_pred
plt.scatter(y_pred, errors)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Prediction Error")
plt.title("Prediction Errors")
plt.show()
