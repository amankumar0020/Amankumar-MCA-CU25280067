import pandas as pd

df = pd.read_csv("data.csv")

# Example mathematical transformation
df["Age_Squared"] = df["Age"] ** 2

print(df)
