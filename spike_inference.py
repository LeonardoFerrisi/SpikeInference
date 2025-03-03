#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import multiprocessing
import scipy.io as sio
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d

# -------------------- Set Environment Variables --------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress unnecessary TensorFlow logs

# Determine the number of CPU cores available
num_cores = multiprocessing.cpu_count()
print("Number of CPU cores available:", num_cores)

# Set CPU parallelism
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_cores)

# -------------------- TensorFlow GPU Setup --------------------
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import (
    Bidirectional, LSTM, Dropout, Dense, LayerNormalization
)
from keras.api.optimizers import Adam
from keras.api.regularizers import l2
from alive_progress import alive_bar

# Check for available GPUs
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from consuming all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found, using CPU.")

# -------------------- Confirm TensorFlow Setup --------------------
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
print("TensorFlow inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())

# =====================================================================================================

def create_sequences(signal, labels, window_size, batch_size=10000):
    """
    Create sequences from the 1D time series signal.
    For each window of LFP data, the label is the spike value at the time immediately after the window.
    """
    num_samples = signal.shape[1] - window_size
    X = np.memmap('X_data.dat', dtype=np.float32, mode='w+', shape=(num_samples, window_size, signal.shape[0]))
    y = np.memmap('y_data.dat', dtype=np.float32, mode='w+', shape=(num_samples,))
    # X, y = [], []
    with alive_bar(num_samples, title="Creating sequences") as bar:
        for i in range(num_samples):
            X[i] = np.transpose(signal[:, i : i + window_size])  # Transpose to match shape (timesteps, channels)
            y[i] = labels[i + window_size]  # Label is the spike immediately after the window
            bar()
    return X, y

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

def main(debug:bool=False):
    # ------------------------------
    # 1. Data Loading and Preprocessing
    # ------------------------------
    #   Load the spikes data
    spikes_electrodes_df = load_data('data/actual_data/electrode.mat', data_key='electrode')
    spikes_1k_df  = load_data('data/actual_data/spikes_1k.mat', data_key='spikes_1k')
    #   **Where spikes_1k_df is a 132x4983702 array, where there are 132 channels and 4983702 time points

    spikes_30k_df        = load_data('data/actual_data/spikes_30k.mat', data_key='spikes_30k')
    spikes_unit_df       = load_data('data/actual_data/unit.mat', data_key='unit')
    spikes_waveform_df   = load_data('data/actual_data/waveform.mat', data_key='waveform')
    sEEG_df              = load_data('data/actual_data/try_sEEG_Data.mat', data_key='Data')

    #   Convert spike_times into a gaussian firing rate ---------------------------
    spikes_times = spikes_1k_df.values[0]

    #   Create logical array of size (1, lfp.shape[1]) of zeros
    ms_buffer = 1000 # 1 s buffer after last timestamp
    spikes = np.zeros(max(spikes_times)+1000)

    #   For each spike time in spike_times, set that index in spikes to 1
    for spike_time in spikes_times:
        spikes[spike_time] = 1
    
    spikes_firing_rate = get_spike_firing_rate(spikes, window_size=100, debug_plot=False)

    #   Handle LFP
    #   Where sEEG_df is a 132x4983702 array, where there are 132 channels and 4983702 time points
    #   Sample Rate of LFP is 1kHz
    #   The time points are in milliseconds

    # Get LFP and spike data
    lfp = sEEG_df.values.astype(np.float32) # convert to float32 for memory efficiency
    spikes = spikes_firing_rate.astype(np.float32) # convert to float32 for memory efficiency

     # Take only a subset of data for initial testing
    # max_samples = 500000  # Start with a smaller dataset for testing
    max_samples = lfp.shape[1]  # Start with a smaller dataset for testing

    num_channels = 2 # Number of channels to use for testing, max of 132

    lfp = sEEG_df.values[:num_channels, :max_samples].astype(np.float32)

    spikes = spikes_firing_rate[:max_samples].astype(np.float32)
            
    # ------------------------------
    # 2. Creating Sequences for the LSTM
    # ------------------------------
    #   Define a window size (number of timesteps per sample)
    window_size = 1  # 50 ms of context
    
    #   Create sequences from the LFP and spike data
    X, y = create_sequences(lfp, spikes, window_size)
    
    # Reshape X to have shape (samples, timesteps, features). In this case, features=1.

    if debug:
        print("Before reshaping:")
        print(f"X shape: {X.shape}, Expected: (num_samples, {window_size}, {num_channels})")
        print(f"y shape: {y.shape}, Expected: (num_samples,)")

    X = X.reshape(-1, window_size, num_channels)

    if debug:
        print("After reshaping:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

   
    # ------------------------------
    # 3. Splitting the Dataset: 70% Training, 30% Validation
    # ------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    if debug:
        print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Validation set shape: X_val={X_val.shape}, y_val={y_val.shape}")
                
    # ------------------------------
    # 4. Building the Bidirectional LSTM Model
    # ------------------------------
    input_timesteps = X_train.shape[1] # Where X is num_samples, timesteps, num_features
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
        epochs=10,         # Adjust the number of epochs as needed
        batch_size=32,     # Adjust batch size as needed
        verbose=1
    )
    time_end_training = time.time()
    print(f"Model training time: {time_end_training - time_start_training} seconds")
    
    # Save the Keras model to an H5 file
    # Create a timestamp string for the model filename
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save model with timestamp in filename
    keras_model_path = f"models/spike_inference_model_{timestamp}.h5"

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
    main(debug=True)
