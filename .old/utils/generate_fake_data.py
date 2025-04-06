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
    
    # Generate the LFP signal as the sum of multiple sine waves
    lfp = np.zeros_like(time)
    for freq in freqs:
        lfp += np.sin(2 * np.pi * freq * time)
    
    # Optionally, you could average the sine waves:
    # lfp = lfp / len(freqs)
    
    # Add Gaussian noise to simulate real-world conditions
    noise_std = 0.5  # Standard deviation of the noise
    lfp += np.random.normal(0, noise_std, size=num_samples)
    
    # Simulate spike events: mark spike=1 when the LFP exceeds a threshold.
    # Adjust the threshold based on the expected amplitude of the LFP.
    spike_threshold = 2.0
    spike = (lfp > spike_threshold).astype(int)
    
    # Create a DataFrame with the time, LFP signal, and spike events
    df = pd.DataFrame({
        'time': time,
        'lfp': lfp,
        'spike': spike
    })
    
    # Write the DataFrame to a CSV file
    output_filename = 'fake_lfp_data.csv'
    df.to_csv(output_filename, index=False)
    print(f"Fake LFP data saved to {output_filename}")

if __name__ == "__main__":
    main()
