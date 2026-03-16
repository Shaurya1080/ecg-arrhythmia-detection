import wfdb
import numpy as np
import os
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

DATASET_PATH = "mitdb"

X = []
y = []

NORMAL = ['N']
ABNORMAL = ['V','A','L','R']

window_size = 180
FS = 360  # MIT-BIH sampling rate

# Bandpass filter (0.5-40 Hz) - removes baseline drift and high-frequency noise
def bandpass(signal, fs=FS, low=0.5, high=40, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

records = os.listdir(DATASET_PATH)
records = [r.split('.')[0] for r in records if '.dat' in r]
records = list(set(records))

print("Records found:", len(records))

for record in records:
    signal, fields = wfdb.rdsamp(DATASET_PATH + "/" + record)
    annotation = wfdb.rdann(DATASET_PATH + "/" + record, 'atr')

    ecg = signal[:, 0]
    # Apply bandpass filter to full signal before extracting beats
    ecg = bandpass(ecg)

    for i, label in enumerate(annotation.symbol):
        pos = annotation.sample[i]
        if pos - window_size < 0 or pos + window_size > len(ecg):
            continue

        beat = ecg[pos - window_size:pos + window_size]

        if label in NORMAL:
            X.append(beat)
            y.append(0)
        elif label in ABNORMAL:
            X.append(beat)
            y.append(1)

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)
print("Labels shape:", y.shape)
print("Class distribution - Normal:", np.sum(y == 0), "| Abnormal:", np.sum(y == 1))

# Normalize
X = (X - np.mean(X)) / np.std(X)

# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train/test split with fixed seed for reproducibility (fixes data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Keep X.npy, y.npy for backward compatibility (full dataset)
np.save("X.npy", X)
np.save("y.npy", y)

print("Data saved (X_train, y_train, X_test, y_test)")
