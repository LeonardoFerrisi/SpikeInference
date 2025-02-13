#!/usr/bin/env python
import numpy as np
import pandas as pd

def main():
    # Parameters for the fake LFP signal
    duration = 10          # Duration in seconds
    sampling_rate = 1000   # Sampling rate in Hz (samples per second)
    num_samples = duration * sampling_rate
    time = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Frequencies to include in the LFP signal
    freqs = [10, 15, 30, 55]
    
    # Generate the LFP signal as the sum of multiple sine waves with random amplitude and phase
    lfp = np.zeros_like(time)
    for freq in freqs:
        amplitude = np.random.uniform(0.5, 1.5)         # Random amplitude between 0.5 and 1.5
        phase = np.random.uniform(0, 2 * np.pi)           # Random phase between 0 and 2pi
        lfp += amplitude * np.sin(2 * np.pi * freq * time + phase)
    
    # Optionally, average the sine waves:
    # lfp = lfp / len(freqs)
    
    # Add Gaussian noise to simulate real-world conditions
    noise_std = np.random.uniform(0.3, 0.7)             # Random noise std between 0.3 and 0.7
    lfp += np.random.normal(0, noise_std, size=num_samples)
    
    # Simulate spike events:
    # Mark spike=1 when the LFP exceeds a threshold that also has a small random variation
    spike_threshold = np.random.uniform(1.8, 2.2)       # Random threshold between 1.8 and 2.2
    spike = (lfp > spike_threshold).astype(int)
    
    # Create a DataFrame with the time, LFP signal, and spike events
    df = pd.DataFrame({
        'time': time,
        'lfp': lfp,
        'spike': spike
    })
    
    # Write the DataFrame to a CSV file
    output_filename = 'fake_lfp_data1.csv'
    df.to_csv(output_filename, index=False)
    print(f"Fake LFP data saved to {output_filename}")

if __name__ == "__main__":
    main()
