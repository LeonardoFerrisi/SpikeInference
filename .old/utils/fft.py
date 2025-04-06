import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv("fake_lfp_data.csv")
# Clean column names if needed
df.columns = df.columns.str.strip().str.lower()  # makes them lowercase and strips whitespace

# Extract time and LFP data
time = df['time']
lfp = df['lfp']

# Calculate the sampling interval (assuming time is uniformly spaced)
dt = np.mean(np.diff(time))
fs = 1 / dt  # sampling frequency

# Compute FFT
n = len(lfp)
fft_result = np.fft.fft(lfp)
freq = np.fft.fftfreq(n, d=dt)

# For plotting, take only the positive frequencies
mask = freq > 0
freq_pos = freq[mask]
fft_amplitude = np.abs(fft_result[mask])

# Plot the FFT amplitude spectrum
plt.figure(figsize=(12, 6))
plt.plot(freq_pos, fft_amplitude, color='green')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of LFP Data")
plt.grid(True)
plt.show()
