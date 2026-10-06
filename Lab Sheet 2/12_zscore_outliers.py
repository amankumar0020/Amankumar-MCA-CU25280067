import pandas as pd
from scipy.stats import zscore

df = pd.read_csv("data.csv")

df["Age_Z"] = zscore(df["Age"].dropna())

outliers = df[df["Age_Z"].abs() > 3]

print(outliers)
