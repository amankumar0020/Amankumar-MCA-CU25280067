
# Q9: Visualize Linear Regression line
exec(open("07_train_linear_regression.py").read())
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test, label="Actual")
plt.scatter(X_test, y_pred, label="Predicted")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
