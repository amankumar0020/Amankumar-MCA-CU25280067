import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

plt.scatter(df.index, df["Age"])
plt.xlabel("Index")
plt.ylabel("Age")
plt.title("Scatter Plot for Detecting Outliers")
plt.show()
