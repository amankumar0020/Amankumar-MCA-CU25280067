"""
Project 1: Handwritten Digit Recognition using a Deep Neural Network.
Dataset: MNIST (60,000 training and 10,000 test images).
Run: python project1_mnist/mnist_dnn.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("results", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)

# Load and normalize the MNIST images.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images into 784 input features.
model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    verbose=1
)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"MNIST test loss: {test_loss:.4f}")
print(f"MNIST test accuracy: {test_accuracy:.4%}")

# Save the trained model.
model.save("trained_models/mnist_digit_model.keras")

# Plot training and validation learning curves.
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST DNN Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/mnist_accuracy.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MNIST DNN Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/mnist_loss.png", dpi=150)
plt.close()

# Show a few predictions for manual inspection.
predictions = np.argmax(model.predict(x_test[:10], verbose=0), axis=1)
print("Actual labels:   ", y_test[:10].tolist())
print("Predicted labels:", predictions.tolist())
