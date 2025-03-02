#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import multiprocessing
import scipy.io as sio
import time


# Set environment variables -----------------------
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Determine the number of CPU cores available
num_cores = multiprocessing.cpu_count()
print("Number of CPU cores available:", num_cores)

# Optionally set environment variables to guide thread usage
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_cores)

import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Bidirectional, LSTM, Dropout, Dense, LayerNormalization
from keras.api.optimizers import Adam
from keras.api.regularizers import l2
from scipy.ndimage import gaussian_filter1d

# Configure TensorFlow to use all CPU cores for both intra- and inter-operation parallelism
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

print("TensorFlow intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
print("TensorFlow inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())

# =====================================================================================================

def create_sequences(signal, labels, window_size):
    """
    Create sequences from the 1D time series signal.
    For each window of LFP data, the label is the spike value at the time immediately after the window.
    """
    X, y = [], []
    for i in range(len(signal) - window_size):
        X.append(signal[i : i + window_size])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

def load_data(data_path:str, data_key:str='Data', debug:bool=True) -> pd.DataFrame:
    """
    Load data from a .mat file and return the data array as a pandas dataframe.

    @param data_path: Path to the .mat file
    @param debug: Print debug information
    @return: Pandas dataframe with the data
    """

    # Load the .mat file (adjust the filename as needed)
    mat_contents = sio.loadmat(data_path)

    # List all variable names in the file
    print(mat_contents.keys())

    # Replace 'data' with the actual variable name stored in your .mat file
    data_array = mat_contents[data_key]

    # Verify the shape and data type
    if debug:
        print("Data loaded from:", data_path)
        print("Shape:", data_array.shape)
        print("Data type:", data_array.dtype)
        print("------------------------------")

    return pd.DataFrame(data_array)

def get_spike_firing_rate(spikes:pd.DataFrame, window_size:int|float, debug_plot:bool=False) -> pd.DataFrame:
    """
    Calculate the spike firing rate from the binary spike array.

    @param spikes: Binary array of spike events (0 or 1)
    @param window_size: Size of the window to use for the convolution
    @return: Spike firing rate array
    """
    std = window_size // 2.0 # std deviation of the gaussian
    spike_firing_rate = gaussian_filter1d(spikes, truncate=2.0, sigma=std)

    if debug_plot: 
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot the spike firing rate
        axs[0].plot(spike_firing_rate, color='blue')
        axs[0].set_title('Spike Firing Rate')
        axs[0].set_ylabel('Firing Rate')
        axs[0].grid()

        # Plot the spikes
        axs[1].plot(spikes, color='red')
        axs[1].set_title('Spikes')
        axs[1].set_xlabel('Time (ms)')
        axs[1].set_ylabel('Spikes')
        axs[1].grid()

        plt.tight_layout()
        plt.show()
    
    return spike_firing_rate

def main():
    # ------------------------------
    # 1. Data Loading and Preprocessing
    # ------------------------------
    # Load the spikes data
    spikes_electrodes_df = load_data('data/actual_data/electrode.mat', data_key='electrode')
    spikes_1k_df  = load_data('data/actual_data/spikes_1k.mat', data_key='spikes_1k')
    # **Where spikes_1k_df is a 132x4983702 array, where there are 132 channels and 4983702 time points

    spikes_30k_df        = load_data('data/actual_data/spikes_30k.mat', data_key='spikes_30k')
    spikes_unit_df       = load_data('data/actual_data/unit.mat', data_key='unit')
    spikes_waveform_df   = load_data('data/actual_data/waveform.mat', data_key='waveform')
    sEEG_df              = load_data('data/actual_data/try_sEEG_Data.mat', data_key='Data')

    # Convert spike_times into a gaussian firing rate ---------------------------
    spikes_times = spikes_1k_df.values[0]

    # Create logical array of size (1, lfp.shape[1]) of zeros
    ms_buffer = 1000 # 1 s buffer after last timestamp
    spikes = np.zeros(max(spikes_times)+1000)

    # For each spike time in spike_times, set that index in spikes to 1
    for spike_time in spikes_times:
        spikes[spike_time] = 1
    
    spikes_firing_rate = get_spike_firing_rate(spikes, window_size=100, debug_plot=False)

    # Handle LFP
    # Where sEEG_df is a 132x4983702 array, where there are 132 channels and 4983702 time points
    # Sample Rate of LFP is 1kHz
    # The time points are in milliseconds

    # Get LFP and spike data
    lfp = sEEG_df.values
        
    # ------------------------------
    # 2. Creating Sequences for the LSTM
    # ------------------------------
    # Define a window size (number of timesteps per sample)
    window_size = 100  # You can adjust this based on your sampling rate & desired context
    
    # Create sequences using the sliding window approach.
    # The label for each sequence is the spike value immediately after the window.

    # For simplicity, we'll use just one channel of LFP data.
    # use just one channel for lfp
    lfp_chan_1 = lfp[0, :]

    X, y = create_sequences(lfp_chan_1, spikes, window_size)
    
    # Reshape X to have shape (samples, timesteps, features). In this case, features=1.
    X = X.reshape(-1, window_size, 1)
    
    print("Data shapes:")
    print("X:", X.shape)
    print("y:", y.shape)
    
    # ------------------------------
    # 3. Splitting the Dataset: 70% Training, 30% Validation
    # ------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Validation set shape: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # ------------------------------
    # 4. Building the Bidirectional LSTM Model
    # ------------------------------
    input_timesteps = X_train.shape[1]
    input_features = X_train.shape[2]
    
    model = Sequential([
        # First Bidirectional LSTM layer; return_sequences=True to allow stacking
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(input_timesteps, input_features)),
        Dropout(0.2),
        
        # Second Bidirectional LSTM layer; return_sequences=False as it's the last LSTM layer
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        
        # Final Dense layer for binary classification (predicting spike or no spike)
        Dense(1, activation='sigmoid')
    ])
    

    time_start_model = time.time()

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()

    time_end_model = time.time()
    print(f"Model building time: {time_end_model - time_start_model} seconds")
    
    # ------------------------------
    # 5. Training the Model
    # ------------------------------

    time_start_training = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,         # Adjust the number of epochs as needed
        batch_size=32,     # Adjust batch size as needed
        verbose=1
    )
    time_end_training = time.time()
    print(f"Model training time: {time_end_training - time_start_training} seconds")
    
    # Save the Keras model to an H5 file
    keras_model_path = "spike_inference_model.h5"
    model.save(keras_model_path)
    
    print(f"Keras model saved to {keras_model_path}")
    # ------------------------------
    # 6. Plotting Training History
    # ------------------------------
    # Plot Training & Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot Training & Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
