import pandas as pd

df = pd.read_csv("data.csv")

# Change "Age" to the numerical column in your dataset
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["Age"] < lower_limit) |
    (df["Age"] > upper_limit)
]

print(outliers)
