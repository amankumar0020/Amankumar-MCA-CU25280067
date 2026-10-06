import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

scaler = MinMaxScaler()
normalized_age = scaler.fit_transform(df[["Age"]])

plt.hist(normalized_age, bins=10)
plt.xlabel("Normalized Age")
plt.ylabel("Frequency")
plt.title("Histogram After Min-Max Normalization")
plt.show()
