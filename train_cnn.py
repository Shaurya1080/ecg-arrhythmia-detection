import numpy as np
import tensorflow as tf
import os

# Load pre-split data (avoids data leakage - test set never seen during training)
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Class weights to handle imbalance (more weight for minority class: abnormal)
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weight = dict(zip(classes, weights))

model = tf.keras.Sequential([

    tf.keras.layers.Conv1D(32,5,activation='relu',input_shape=(360,1)),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(64,5,activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model.compile(

    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']

)

model.summary()

os.makedirs("models", exist_ok=True)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight
)

model.save("models/ecg_cnn.h5")

print("Model saved")
