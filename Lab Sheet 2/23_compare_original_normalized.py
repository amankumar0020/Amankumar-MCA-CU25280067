import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data.csv")
df_compare = df.copy()

scaler = MinMaxScaler()
df_compare["Age_Normalized"] = scaler.fit_transform(
    df_compare[["Age"]]
)

print(df_compare[["Age", "Age_Normalized"]].head(10))
