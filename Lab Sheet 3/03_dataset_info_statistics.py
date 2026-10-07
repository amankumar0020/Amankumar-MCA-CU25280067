
# Q3: Explore dataset information and descriptive statistics
exec(open("01_load_dataset.py").read())
print("Dataset Information:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())
