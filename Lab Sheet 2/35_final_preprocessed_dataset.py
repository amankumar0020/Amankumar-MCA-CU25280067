import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
df_final = df.copy()

# Fill numerical missing values with median
numeric_columns = df_final.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df_final[column] = df_final[column].fillna(
        df_final[column].median()
    )

# Fill categorical missing values with mode
categorical_columns = df_final.select_dtypes(include="object").columns

for column in categorical_columns:
    if not df_final[column].mode().empty:
        df_final[column] = df_final[column].fillna(
            df_final[column].mode()[0]
        )

# Remove duplicate records
df_final = df_final.drop_duplicates()

# Save final dataset
df_final.to_csv(
    "final_preprocessed_dataset.csv",
    index=False
)

print("Final Preprocessed Dataset:")
print(df_final.head())

print("\nMissing values:")
print(df_final.isnull().sum())

print("\nDataset shape:")
print(df_final.shape)

print("\nFinal dataset saved successfully.")
