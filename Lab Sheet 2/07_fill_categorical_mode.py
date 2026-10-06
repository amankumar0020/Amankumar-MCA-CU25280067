import pandas as pd

df = pd.read_csv("data.csv")
df_clean = df.copy()

categorical_columns = df_clean.select_dtypes(include="object").columns

for column in categorical_columns:
    if not df_clean[column].mode().empty:
        df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])

print(df_clean)
