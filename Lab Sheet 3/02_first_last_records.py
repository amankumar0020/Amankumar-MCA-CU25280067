
# Q2: Display first and last five records
exec(open("01_load_dataset.py").read())
print("First 5 records:")
print(df.head())
print("\nLast 5 records:")
print(df.tail())
