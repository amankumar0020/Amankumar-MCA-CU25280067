import pandas as pd

df = pd.read_csv("data.csv")
df_backward = df.copy()

df_backward = df_backward.bfill()

print(df_backward)
