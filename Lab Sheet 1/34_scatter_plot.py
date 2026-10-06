import matplotlib.pyplot as plt

plt.scatter(
    df["sepal length (cm)"],
    df["petal length (cm)"]
)

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Scatter Plot")

plt.show()
