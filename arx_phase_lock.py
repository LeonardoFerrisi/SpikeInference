import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from scipy.signal import correlate, hilbert

# ------------------------------
# 1. Data Loading and Preprocessing
# ------------------------------
data_dir = "./data"
filename = "fake_lfp_data.csv"  # Change as needed
data_path = os.path.join(data_dir, filename)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file {data_path} not found.")

# Read CSV file: expected columns 'time', 'lfp', 'spike'
df = pd.read_csv(data_path)
lfp = df['lfp'].values         # continuous LFP signal
spike = df['spike'].values     # binary spike train (0/1)

# ------------------------------
# 2. Build an ARX model: Predict spikes from lagged LFP
# ------------------------------
lag_order = 50

def create_arx_matrix(signal, lag):
    X = []
    for i in range(lag, len(signal)):
        X.append(signal[i-lag:i])
    return np.array(X)

# Create predictor matrix and target vector
X_arx = create_arx_matrix(lfp, lag_order)
y_arx = spike[lag_order:]  # corresponding spike values

# Fit a simple linear regression as an ARX model
arx_model = LinearRegression()
arx_model.fit(X_arx, y_arx)
y_pred = arx_model.predict(X_arx)

# ------------------------------
# 3. Evaluation Metrics: Loss Functions and Accuracy
# ------------------------------
# For binary classification, threshold predictions at 0.5
y_pred_binary = (y_pred >= 0.5).astype(int)

# Classification accuracy: fraction of correct binary predictions
accuracy = accuracy_score(y_arx, y_pred_binary)

# Mean Squared Error (MSE): continuous prediction error
mse = mean_squared_error(y_arx, y_pred)

# Binary Cross-Entropy Loss (Log Loss): loss when predictions are interpreted as probabilities
bce_loss = log_loss(y_arx, y_pred)

print("Evaluation Metrics:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Mean Squared Error: {mse:.3f}")
print(f"  Binary Cross-Entropy Loss: {bce_loss:.3f}")

# ------------------------------
# 4. Augment with Phase-Locking Cross-Correlation
# ------------------------------
analytic_signal = hilbert(lfp)
instant_phase = np.angle(analytic_signal)
phase_cos = np.cos(instant_phase)
phase_sin = np.sin(instant_phase)

corr_cos = correlate(spike, phase_cos, mode='same')
corr_sin = correlate(spike, phase_sin, mode='same')

lags = np.arange(-len(spike)//2, len(spike)//2)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(lags, corr_cos, color='blue')
plt.title('Cross-Correlation: Spike vs cos(Phase)')
plt.xlabel('Lag (samples)')
plt.ylabel('Correlation')

plt.subplot(1,2,2)
plt.plot(lags, corr_sin, color='green')
plt.title('Cross-Correlation: Spike vs sin(Phase)')
plt.xlabel('Lag (samples)')
plt.ylabel('Correlation')
plt.tight_layout()
plt.show()
