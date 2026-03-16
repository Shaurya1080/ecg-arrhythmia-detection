import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/ecg_cnn.h5")

# Use held-out test set only (no data leakage)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)
