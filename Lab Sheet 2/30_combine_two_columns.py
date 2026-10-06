import pandas as pd

df = pd.read_csv("data.csv")

# Example: combine First_Name and Last_Name
df["Full_Name"] = (
    df["First_Name"] + " " + df["Last_Name"]
)

print(df)
