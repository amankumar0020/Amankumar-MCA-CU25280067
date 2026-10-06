import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

sns.boxplot(x=df["Age"])
plt.title("Box Plot of Age")
plt.show()
