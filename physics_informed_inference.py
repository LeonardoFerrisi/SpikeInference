import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Step 1: Define the LFP Kernel Function
# -------------------------------
def lfp_kernel(t, tau=20.0):
    """
    Computes a decaying exponential kernel.
    
    Parameters:
        t (numpy array): time values (ms)
        tau (float): time constant (ms)
    
    Returns:
        numpy array: kernel values computed as exp(-t/tau) for t>=0, 0 otherwise.
    """
    # Ensure that the kernel is causal (t >= 0)
    kernel = np.where(t >= 0, np.exp(-t / tau), 0)
    return kernel

# Define kernel parameters
kernel_length_ms = 100  # length of kernel in ms (adjust as needed)
dt               = 1.0  # time step in ms (assuming data is in ms)
t_vals           = np.arange(0, kernel_length_ms, dt)
kernel_vals      = lfp_kernel(t_vals, tau=20.0)  # 20 ms time constant

# Convert kernel to PyTorch tensor and reshape to (1, 1, kernel_length) for conv1d
kernel_tensor    = torch.tensor(kernel_vals, dtype=torch.float32).view(1, 1, -1)

# -------------------------------
# Step 2: Define the Forward Model (Convolution)
# -------------------------------
# In our approach, the forward model is given by:
#    LFP(t) = sum_i K(t - t_i) + noise
# where the kernel K is defined above.
# Later, we use this idea to define a physics-informed loss.

# -------------------------------
# Step 3: Create a PyTorch Dataset to Load CSV Files
# -------------------------------
class LFPSPDataset(Dataset):
    """
    A PyTorch Dataset to load LFP and spike data from CSV files in a folder.
    
    Each CSV file is assumed to have columns:
        time, LFP_time_series, Spikes
        
    The data is returned as tensors of shape (1, time_steps).
    """
    def __init__(self, data_folder):
        super(LFPSPDataset, self).__init__()
        # Get list of all CSV files in the data_folder
        self.files = glob.glob(os.path.join(data_folder, '*.csv'))
        
        # Load all data into lists (or you can load on the fly)
        self.data = []
        for file in self.files:
            df = pd.read_csv(file)
            # Assuming the CSV columns are named exactly 'time', 'LFP_time_series', 'Spikes'
            # We only need the LFP and spike columns (time is not used for training here)
            lfp = df['lfp'].values.astype(np.float32)
            spikes = df['spike'].values.astype(np.float32)
            # Optionally, you could normalize the LFP if needed
            # For training, reshape to (1, time_steps)
            lfp = np.expand_dims(lfp, axis=0)
            spikes = np.expand_dims(spikes, axis=0)
            self.data.append((lfp, spikes))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        lfp, spikes = self.data[idx]
        # Convert numpy arrays to torch tensors
        lfp_tensor = torch.tensor(lfp, dtype=torch.float32)
        spikes_tensor = torch.tensor(spikes, dtype=torch.float32)
        return lfp_tensor, spikes_tensor

# Create dataset and DataLoader
data_folder     = 'data/'  # folder where CSV files are stored
dataset         = LFPSPDataset(data_folder)
batch_size      = 4  # adjust batch size as needed
training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# Step 4: Define the Inversion Network (CNN) Architecture
# -------------------------------
class LFPtoSpikeNet(nn.Module):
    def __init__(self):
        super(LFPtoSpikeNet, self).__init__()
        # A simple CNN with two 1D convolutional layers.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, padding=2)
        # Sigmoid activation to produce spike probabilities (between 0 and 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch_size, 1, time_steps)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x

# Instantiate the model and set it to training mode
model = LFPtoSpikeNet()
model.train()

# -------------------------------
# Step 5: Define the Physics-Informed Loss Function and Training Loop
# -------------------------------
def physics_loss(predicted_spikes, observed_lfp, kernel_tensor):
    """
    Computes the physics loss: the mean squared error between the observed LFP and the
    reconstructed LFP obtained by convolving the predicted spike train with the kernel.
    
    Parameters:
        predicted_spikes (torch.Tensor): shape (batch_size, 1, time_steps)
        observed_lfp (torch.Tensor): shape (batch_size, 1, time_steps)
        kernel_tensor (torch.Tensor): shape (1, 1, kernel_length)
    
    Returns:
        torch.Tensor: scalar loss value
    """
    # Convolve predicted spike probabilities with the fixed kernel.
    # Use padding to keep the output the same size as the input.
    recon_lfp = nn.functional.conv1d(predicted_spikes, kernel_tensor, padding=kernel_tensor.shape[-1]//2)
    # Remove the last time step to match the desired output shape
    recon_lfp = recon_lfp[:, :, :-1]
    # Compute mean squared error between reconstructed LFP and observed LFP
    loss = torch.mean((recon_lfp - observed_lfp) ** 2)
    return loss

# Define the optimizer (using Adam)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loss weighting factors for spike loss and physics loss
lambda1   = 1.0  # weight for spike prediction loss
lambda2   = 1.0  # weight for physics (reconstruction) loss

# Training loop
num_epochs = 5000  # adjust as needed
for epoch in range(num_epochs):
    epoch_loss = 0.0  # accumulator for total loss
    for batch_idx, (lfp_input, true_spikes) in enumerate(training_loader):
        # lfp_input and true_spikes have shape (batch_size, 1, time_steps)
        
        # Zero the gradients from the previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute the predicted spike probabilities from the LFP input
        predicted_spikes = model(lfp_input)
        
        # Compute spike prediction loss (e.g., mean squared error)
        spike_loss = nn.functional.mse_loss(predicted_spikes, true_spikes)
        
        # Compute physics loss: how well the predicted spikes, when convolved with the kernel,
        # reconstruct the observed LFP.
        phys_loss = physics_loss(predicted_spikes, lfp_input, kernel_tensor)
        
        # Total loss is a weighted sum of the spike loss and the physics loss.
        total_loss = lambda1 * spike_loss + lambda2 * phys_loss
        
        # Backward pass: compute gradients
        total_loss.backward()
        
        # Update model parameters using the optimizer
        optimizer.step()
        
        # Accumulate loss (multiply by batch size for averaging later)
        epoch_loss += total_loss.item() * lfp_input.size(0)
    
    # Calculate average loss for this epoch
    avg_loss = epoch_loss / len(training_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# -------------------------------
# End of Script
# -------------------------------
