import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

df["Age_Log"] = np.log1p(df["Age"])

print(df[["Age", "Age_Log"]])
