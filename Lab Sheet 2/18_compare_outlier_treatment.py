import pandas as pd

df = pd.read_csv("data.csv")

print("Original dataset:")
print(df["Age"].describe())

Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_treated = df[
    (df["Age"] >= lower) &
    (df["Age"] <= upper)
]

print("\nAfter outlier treatment:")
print(df_treated["Age"].describe())
