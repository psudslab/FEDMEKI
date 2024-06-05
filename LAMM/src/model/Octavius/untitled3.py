#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:23:06 2024

@author: xmw5190
"""

import wfdb
import matplotlib.pyplot as plt

record = wfdb.rdrecord("/data/xiaochen/data/physionet.org/files/apnea-ecg/1.0.0/a01e")
# record = wfdb.rdrecord("/data/xiaochen/data/physionet.org/files/ptb-xl/1.0.3/records100/00000/00001_lr")
signals = record.p_signal
print(record.p_signal.shape)


annotations = wfdb.rdann("/data/xiaochen/data/physionet.org/files/slpdb/1.0.0/slp01a", 'st')
print("Annotation symbols:", len(annotations.symbol))
# print("Annotation sample indices:", annotation.sample)
plt.figure(figsize=(10, 4))
plt.plot(signals[:,0])
plt.title('ECG Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()







import numpy as np

window_size = 200  # 200 samples before and after the annotation

segments = []
labels = []

# Loop through the annotations
for i in range(len(annotations.sample)):
    pos = annotations.sample[i]
    start = pos - window_size
    end = pos + window_size

    # Check if the window is valid
    if start >= 0 and end < len(signals):
        segment = signals[start:end, :]
        segments.append(segment)
        labels.append(annotations.symbol[i])  # Use i to access the corresponding symbol

segments = np.array(segments)
labels = np.array(labels)

print(segments.shape)
print([i for i in labels if i != 'N'])
