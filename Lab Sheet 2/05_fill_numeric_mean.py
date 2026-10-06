import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
df_clean = df.copy()

numeric_columns = df_clean.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())

print(df_clean)
