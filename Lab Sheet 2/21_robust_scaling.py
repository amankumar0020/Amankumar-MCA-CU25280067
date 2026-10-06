import pandas as pd
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("data.csv")
df_scaled = df.copy()

scaler = RobustScaler()
df_scaled[["Age"]] = scaler.fit_transform(df_scaled[["Age"]])

print(df_scaled)
