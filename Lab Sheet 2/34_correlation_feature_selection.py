import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

correlation = df.corr(numeric_only=True)

print(correlation)

plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True)
plt.title("Correlation Matrix")
plt.show()
