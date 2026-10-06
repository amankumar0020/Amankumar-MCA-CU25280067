df_sorted = df.sort_values(by="Age")
print(df_sorted)

# Descending order:
df_sorted_desc = df.sort_values(by="Age", ascending=False)
print(df_sorted_desc)
