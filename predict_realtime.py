import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/ecg_cnn.h5")

def predict_ecg(ecg_signal):

    ecg_signal = np.array(ecg_signal)

    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    ecg_signal = ecg_signal.reshape(1,360,1)

    prediction = model.predict(ecg_signal)

    return prediction[0][0]
