import pandas as pd

df = pd.read_csv("data.csv")

missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)
