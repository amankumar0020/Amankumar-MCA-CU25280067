# Install first:
# pip install category_encoders

import pandas as pd
import category_encoders as ce

df = pd.read_csv("data.csv")

# Change "Gender" to the categorical column in your dataset
encoder = ce.BinaryEncoder(cols=["Gender"])

df_binary = encoder.fit_transform(df)

print(df_binary)
