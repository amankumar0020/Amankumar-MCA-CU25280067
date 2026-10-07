
# Q12: Predict for new user-defined input
exec(open("07_train_linear_regression.py").read())
new_data = [[5.0]]
print("Predicted House Price:", model.predict(new_data)[0])
