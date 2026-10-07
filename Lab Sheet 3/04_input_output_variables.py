
# Q4: Identify independent and dependent variables
exec(open("01_load_dataset.py").read())
X = df.drop("Price", axis=1)
y = df["Price"]
print("Input variables:", list(X.columns))
print("Output variable: Price")
