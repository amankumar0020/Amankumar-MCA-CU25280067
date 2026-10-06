import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler
)

df = pd.read_csv("data.csv")
data = df[["Age"]].dropna()

minmax = MinMaxScaler().fit_transform(data)
standard = StandardScaler().fit_transform(data)
robust = RobustScaler().fit_transform(data)
maxabs = MaxAbsScaler().fit_transform(data)

comparison = pd.DataFrame({
    "Original": data["Age"].values,
    "MinMax": minmax.flatten(),
    "Standard": standard.flatten(),
    "Robust": robust.flatten(),
    "MaxAbs": maxabs.flatten()
})

print(comparison.head(10))
