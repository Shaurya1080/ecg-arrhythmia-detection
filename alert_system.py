# Suppress TensorFlow verbose logs (oneDNN, CPU features, etc.)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=errors only
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

from predict_realtime import predict_ecg
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

DATASET_PATH = "mitdb"
NORMAL = ['N']
ABNORMAL = ['V', 'A', 'L', 'R']
WINDOW_SIZE = 180
FS = 360

def bandpass(signal, fs=FS, low=0.5, high=40, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

# Get all records from MIT-BIH database
records = [r.split('.')[0] for r in os.listdir(DATASET_PATH) if '.dat' in r]
records = list(set(records))

# Pick a random record
record = records[np.random.randint(0, len(records))]

# Load raw ECG signal and annotations from MIT-BIH
signal, fields = wfdb.rdsamp(os.path.join(DATASET_PATH, record))
annotation = wfdb.rdann(os.path.join(DATASET_PATH, record), 'atr')

ecg = signal[:, 0]  # Channel 0 (MLII)
ecg = bandpass(ecg)  # Match training preprocessing

# Collect valid beat segments with labels
beats = []
labels = []
for i, symbol in enumerate(annotation.symbol):
    pos = annotation.sample[i]
    if pos - WINDOW_SIZE < 0 or pos + WINDOW_SIZE > len(ecg):
        continue
    if symbol in NORMAL:
        beats.append(ecg[pos - WINDOW_SIZE:pos + WINDOW_SIZE])
        labels.append(0)
    elif symbol in ABNORMAL:
        beats.append(ecg[pos - WINDOW_SIZE:pos + WINDOW_SIZE])
        labels.append(1)

# Pick a random beat from this record
sample_idx = np.random.randint(0, len(beats))
ecg_segment = np.array(beats[sample_idx])
true_label = labels[sample_idx]

result = predict_ecg(ecg_segment)

if result > 0.5:
    print("ALERT: Abnormal ECG Detected")
else:
    print("Normal ECG")

print(f"  Record: {record} | Prediction: {result:.3f} | Actual: {'Abnormal' if true_label == 1 else 'Normal'}")
