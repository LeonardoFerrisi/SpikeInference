#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense, LayerNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
import time

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

def main():
    # ------------------------------
    # 1. Data Loading and Preprocessing
    # ------------------------------
    # Change the data_dir and filename as needed.
    data_dir = "./data"
    filename = "fake_lfp_data.csv"  # CSV file generated previously
    data_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please ensure the CSV exists.")
        return

    # Read the CSV file; expecting columns: 'time', 'lfp', and 'spike'
    df = pd.read_csv(data_path)
    
    # Extract the LFP and spike columns (assuming LFP is our feature and spike is our label)
    lfp = df['lfp'].values
    spike = df['spike'].values
    
    # ------------------------------
    # 2. Creating Sequences for the LSTM
    # ------------------------------
    # Define a window size (number of timesteps per sample)
    window_size = 100  # You can adjust this based on your sampling rate & desired context
    
    # Create sequences using the sliding window approach.
    # The label for each sequence is the spike value immediately after the window.
    X, y = create_sequences(lfp, spike, window_size)
    
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
        epochs=50,         # Adjust the number of epochs as needed
        batch_size=64,     # Adjust batch size as needed
        verbose=1
    )
    
    time_end_training = time.time()
    print(f"Model training time: {time_end_training - time_start_training} seconds")

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
