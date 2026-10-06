import matplotlib.pyplot as plt

plt.hist(df["sepal length (cm)"], bins=10)

plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Length")

plt.show()
