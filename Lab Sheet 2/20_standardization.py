import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
df_scaled = df.copy()

scaler = StandardScaler()
df_scaled[["Age"]] = scaler.fit_transform(df_scaled[["Age"]])

print(df_scaled)
