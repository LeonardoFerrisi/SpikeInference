import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv("fake_lfp_data.csv")

# Clean the column names (if necessary)
df.columns = df.columns.str.strip().str.lower()

# Verify columns
print("Columns:", df.columns.tolist())

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the lfp_data over time
plt.plot(df['time'], df['lfp'], label='LFP Data', color='blue')

# Overlay spike events (assuming spike is stored as a boolean or 0/1)
spike_mask = df['spike'] == True  # or df['spike'] == 1 if spikes are marked as 1
plt.scatter(df.loc[spike_mask, 'time'], df.loc[spike_mask, 'lfp'],
            color='red', label='Spike', zorder=5)

# Label the axes and add a title
plt.xlabel("Time")
plt.ylabel("LFP Data")
plt.title("LFP Data with Spike Events")
plt.legend()

# Show the plot
plt.show()
