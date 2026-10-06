import pandas as pd

df = pd.read_csv("data.csv")
df_capped = df.copy()

lower = df_capped["Age"].quantile(0.01)
upper = df_capped["Age"].quantile(0.99)

df_capped["Age"] = df_capped["Age"].clip(
    lower=lower,
    upper=upper
)

print(df_capped)
