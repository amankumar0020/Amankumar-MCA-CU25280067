import seaborn as sns
import matplotlib.pyplot as plt

correlation = df.corr(numeric_only=True)

print(correlation)

sns.heatmap(correlation, annot=True)

plt.title("Correlation Matrix Heatmap")
plt.show()
