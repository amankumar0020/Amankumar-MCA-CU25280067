
# Q16: Compare actual and predicted values graphically
exec(open("13_multiple_linear_regression.py").read())
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
