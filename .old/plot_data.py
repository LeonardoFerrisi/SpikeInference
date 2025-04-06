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
from alive_progress import alive_bar


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
        print("Shape:", data_array.shape)
        print("Data type:", data_array.dtype)

    return pd.DataFrame(data_array)

def main():
    # ------------------------------
    # 1. Data Loading and Preprocessing
    # ------------------------------

    # Read the CSV file; expecting columns: 'time', 'lfp', and 'spike'
    sEEG_df = load_data('data/actual_data/try_sEEG_Data.mat', data_key='Data')

    # ------------------------------
    # 2. Plotting
    # ------------------------------

    
    # Get the number of channels and time samples
    sample_rate  = 1000 # 1kHz
    # num_channels = sEEG_df.shape[0]    # Should be 132
    num_channels = 100 #sEEG_df.shape[0]    # Should be 132

    time_samples = sEEG_df.shape[1]      # ~4,983,702

    # Convert time_samples from current timescale into ms
    time_samples = np.arange(time_samples) / sample_rate
    offset = 300
    fig, ax = plt.subplots(figsize=(15, 20))
    with alive_bar(num_channels, title='Plotting Channels') as bar:
        for i in range(num_channels):
            # Downsample the data for channel i
            signal = sEEG_df.iloc[i, :].values / 100000
            # Optionally, center the signal so variations are visible
            signal_centered = signal - np.mean(signal)
            # Offset the signal so that its baseline is at channel number (i+1)
            ax.plot(time_samples, signal_centered + (i + offset), label=f'Channel {i+1}')
            bar()
    ax.legend(loc='upper right')

    ax.set_xlabel('Time Samples (Downsampled)')
    ax.set_ylabel('Channel')
    ax.set_title('sEEG Data: Waveforms for 132 Channels')
    # ax.set_yticks(np.arange(1, num_channels + 1))
    plt.show()

if __name__ == "__main__":
    main()