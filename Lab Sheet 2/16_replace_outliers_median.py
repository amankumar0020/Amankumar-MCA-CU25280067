import pandas as pd

df = pd.read_csv("data.csv")
df_median = df.copy()

Q1 = df_median["Age"].quantile(0.25)
Q3 = df_median["Age"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
median_value = df_median["Age"].median()

df_median.loc[
    (df_median["Age"] < lower_limit) |
    (df_median["Age"] > upper_limit),
    "Age"
] = median_value

print(df_median)
