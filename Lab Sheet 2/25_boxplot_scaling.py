import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

scaler = StandardScaler()
scaled_age = scaler.fit_transform(df[["Age"]])

sns.boxplot(x=scaled_age.flatten())
plt.title("Box Plot After Standard Scaling")
plt.show()
