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

spikes_file = 'data/actual_data/patient1/spike_times_set1_1k.mat'
spikes_key  = 'spike_times_set1_1k'

spikes_1k_df  = load_data(spikes_file, data_key=spikes_key)

#   Convert spike_times into a gaussian firing rate ---------------------------
spikes_times  = spikes_1k_df.values[0]

#   Create logical array of size (1, lfp.shape[1]) of zeros
ms_buffer     = 1000 # 1s buffer after last timestamp

spikes_times  = np.round(spikes_times).astype(np.int32) # Convert to int for indexing
spikes  = np.zeros(max(spikes_times) + ms_buffer)
spikes_firing_rate = get_spike_firing_rate(spikes, window_size=10000, debug_plot=True)