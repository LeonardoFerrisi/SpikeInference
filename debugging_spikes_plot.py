import os
import numpy as np
import pandas as pd
import multiprocessing
import scipy.io as sio
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def load_data(data_path:str, data_key:str='Data', debug:bool=True) -> pd.DataFrame:
    """
    Load data from a .mat file and return the data array as a pandas dataframe.

    @param data_path: Path to the .mat file
    @param debug: Print debug information
    @return: Pandas dataframe with the data
    """

    # Load the .mat file
    mat_contents = sio.loadmat(data_path)

    # List all variable names in the file
    print(mat_contents.keys())

    # Get data from specified key
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
    @param window_size: Size of the window to use for the convolution in ms
    @return: Spike firing rate array
    """
    std = window_size // 2.0 # std deviation of the gaussian
    spike_firing_rate = gaussian_filter1d(spikes, truncate=2.0, sigma=std)

    if debug_plot: 
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Create time axis in seconds
        time_sec = np.arange(len(spike_firing_rate)) / 1000  # convert ms to seconds
        
        # Plot the spike firing rate
        axs[0].plot(time_sec, spike_firing_rate, color='blue')
        axs[0].set_title('Spike Firing Rate')
        axs[0].set_ylabel('Firing Rate')
        axs[0].grid()

        # Plot the spikes
        axs[1].plot(time_sec, spikes, color='red')
        axs[1].set_title('Spikes')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Spikes')
        axs[1].grid()

        plt.tight_layout()
        plt.show()
    
    return spike_firing_rate

def debugging_spikes_plot(samples='all', lfp_channel=1):
    """
    @params samples: Number of samples to plot. Set to 'all' to plot all samples.
    """

    spikes_file = 'data/actual_data/patient1/spike_times_set1_1k.mat'
    spikes_key  = 'spike_times_set1_1k'

    lfp_file      = 'data/actual_data/patient1/try_sEEG_Data.mat'
    lfp_key       ='Data'

    sEEG_df       = load_data(lfp_file, data_key=lfp_key)
    spikes_1k_df  = load_data(spikes_file, data_key=spikes_key)

    if samples == 'all':
        max_samples = sEEG_df.shape[1] 
    else:
        if type(samples) != int:
            raise ValueError("max_samples must be an integer or 'all'")
        else:
            max_samples = min(samples, sEEG_df.shape[1])


    # Standardize each channel (axis=1) using StandardScaler:
    # Initialize 1D array for the single channel
    lfp = sEEG_df.values[lfp_channel, :max_samples].astype(np.float32)

    #   Convert spike_times into a gaussian firing rate ---------------------------
    spikes_times  = spikes_1k_df.values[0]

    #   Create logical array of size (1, lfp.shape[1]) of zeros
    ms_buffer     = 1000 # 1s buffer after last timestamp

    spikes_times  = np.round(spikes_times).astype(np.int32) # Convert to int for indexing

    spikes = np.zeros(len(sEEG_df.values[0]))

    #   For each spike time in spike_times, set that index in spikes to 1
    for spike_time in spikes_times:
        if spike_time < len(spikes):  # Ensure we don't go out of bounds
            spikes[spike_time] = 1

    spikes_firing_rate = get_spike_firing_rate(spikes, window_size=20, debug_plot=False)

    # Create time axis in seconds (1kHz sampling rate means each sample is 1ms)
    time_sec = np.arange(min(len(lfp), len(spikes_firing_rate))) / 1000

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Create time axis in milliseconds (1kHz sampling rate means each sample is 1ms)
    time_ms = np.arange(min(len(lfp), len(spikes_firing_rate)))

    # Plot the spike firing rate on top subplot
    axs[0].plot(time_ms, spikes_firing_rate[:len(time_ms)], color='red')
    axs[0].set_title('Spike Firing Rate')
    axs[0].set_ylabel('Firing Rate')
    axs[0].set_xlabel('Time (ms)')
    axs[0].grid(True)

    # Plot the LFP data on bottom subplot
    # Only plot up to the minimum length
    min_length = min(len(lfp), len(spikes_firing_rate))
    axs[1].plot(time_ms, lfp[:min_length], color='blue')
    axs[1].set_title('Local Field Potential (LFP)')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    debugging_spikes_plot(samples=250000, lfp_channel=65)