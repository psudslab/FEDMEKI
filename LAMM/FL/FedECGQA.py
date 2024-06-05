#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:49:33 2024

@author: xmw5190
"""




import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Function to apply Butterworth low-pass filter



def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency must be between 0 and 1
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Read the WFDB record and the annotation file
record = wfdb.rdrecord('/data/xiaochen/data/physionet.org/files/ptb-xl/1.0.3/records100/00000/00800_lr')
# annotation = wfdb.rdann('/data/xiaochen/data/physionet.org/files/ptb-xl/1.0.3/records100/00000/', 'atr')

# Extract the first channel signal
ecg_signal = record.p_signal[:,0]

# Sampling frequency
fs = record.fs

print((record.p_signal))
# Apply a low-pass filter (cutoff at 50 Hz, order 5)
filtered_ecg_signal = butter_lowpass_filter(ecg_signal, 50, fs, 5)
# filtered_ecg_signal = butter_lowpass_filter(ecg_signal, 50, fs, 5)

# Plot the original and filtered signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg_signal)
plt.title('Original ECG Signal')
plt.subplot(2, 1, 2)
plt.plot(filtered_ecg_signal, color='red')
plt.title('Filtered ECG Signal')
plt.tight_layout()
plt.show()
