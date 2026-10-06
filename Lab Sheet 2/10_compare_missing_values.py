import pandas as pd

df = pd.read_csv("data.csv")

print("Missing values before:")
print(df.isnull().sum())

df_clean = df.copy()
df_clean = df_clean.ffill().bfill()

print("\nMissing values after:")
print(df_clean.isnull().sum())
