
# Q17: Analyze effect of each independent variable
exec(open("13_multiple_linear_regression.py").read())
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coefficients)
