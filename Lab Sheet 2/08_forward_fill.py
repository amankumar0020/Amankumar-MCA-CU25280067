import pandas as pd

df = pd.read_csv("data.csv")
df_forward = df.copy()

df_forward = df_forward.ffill()

print(df_forward)
