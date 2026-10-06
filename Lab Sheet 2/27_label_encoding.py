import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")
df_encoded = df.copy()

encoder = LabelEncoder()

# Change "Gender" to the categorical column in your dataset
df_encoded["Gender_Encoded"] = encoder.fit_transform(
    df_encoded["Gender"]
)

print(df_encoded)
