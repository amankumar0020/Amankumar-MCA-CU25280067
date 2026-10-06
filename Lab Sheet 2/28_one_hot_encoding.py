import pandas as pd

df = pd.read_csv("data.csv")

# Change "Gender" to the categorical column in your dataset
df_encoded = pd.get_dummies(
    df,
    columns=["Gender"]
)

print(df_encoded)
