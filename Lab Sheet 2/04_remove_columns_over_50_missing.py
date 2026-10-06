import pandas as pd

df = pd.read_csv("data.csv")

missing_percentage = (df.isnull().sum() / len(df)) * 100
columns_to_remove = missing_percentage[missing_percentage > 50].index

df_clean = df.drop(columns=columns_to_remove)

print(df_clean)
