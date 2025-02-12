import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split  # For a simple 70/30 split
# Can also use GroupShuffleSplit if splitting participants into groups

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dropout, Dense, LayerNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

# ------------------------------
# 1. Data Loading and Preprocessing
# ------------------------------

# Define the directory where your 31 datafiles are stored.
# (Adjust the path and file extension as needed.)
data_dir = "./data"
# TODO: Remove when we know what the file structure looks like

breakpoint()
print("WE DONT KNOW WHAT THE FILE STRUCTURE LOOKS LIKE")

file_count = 30
data_files = glob.glob(os.path.join(data_dir, "*.tbd"))
assert len(data_files) == file_count, f"Expected {file_count} data files but found {len(data_files)}."

# Placeholder lists to store features and labels.
# Here we assume each .npy file contains a dictionary with keys 'X' (features) and 'y' (labels).
# For example, each file could have been saved via:
#   np.save(file_path, {'X': X_data, 'y': y_data})
X_list = []
y_list = []

# Loop through each file and load the data
for file_path in data_files:
    # Load the data assuming each file contains a dictionary.
    # Adjust this if your files are stored differently.
    data = np.load(file_path, allow_pickle=True).item()
    
    # Append features and labels to the lists.
    X_list.append(data['X'])
    y_list.append(data['y'])

# Concatenate data from all files along the sample axis.
# Make sure that the arrays you are concatenating have consistent dimensions.
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

# ------------------------------
# 2. Splitting the Dataset: 70% Training, 30% Validation
# ------------------------------

# Using sklearn's train_test_split for a simple random split.
# If you need stratification or grouping (e.g., by participant), additional parameters or a different splitter may be required.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")

# ------------------------------
# 3. Building the LSTM Model
# ------------------------------

# Determine input shape from your training data.
# Assuming X_train has shape (samples, timesteps, features)
input_timesteps = X_train.shape[1]
input_features = X_train.shape[2]

# Define a simple LSTM model
model = Sequential([
    # First LSTM layer with return_sequences=True if you want to stack LSTMs
    LSTM(64, input_shape=(input_timesteps, input_features), return_sequences=True),
    Dropout(0.2),
    # Second LSTM layer; set return_sequences=False if this is the final LSTM layer
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    # Final Dense layer for binary classification (adjust activation/loss as needed)
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()


#------------
#Daniel's SHitty alt model
#--------------

# Define the model
model = Sequential([
    
    Bidirectional(  
                    LSTM(128, return_sequences = True, recurrent_dropout = 0.2, kernel_regularizer = l2(0.01)), 
                    input_shape=(X_train.shape[1], X_train.shape[2])
                ),
    LayerNormalization(),
    Dropout(0.2),

    Bidirectional( 
                    LSTM(64, return_sequences = True, recurrent_dropout = 0.2, kernel_regularizer = l2(0.01))
                ),
    LayerNormalization(),
    Dropout(0.2),

    Bidirectional(
                    LSTM(32, return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l2(0.01))
                ),
    LayerNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# Display model summary
model.summary()


# ------------------------------
# 4. Training the Model
# ------------------------------

# Fit the model on the training data and validate on the 30% hold-out.
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,         # Set the number of epochs according to your experiment needs.
    batch_size=64,     # Adjust batch size as needed.
    verbose=1
)


#-------------
#plot stuff
#-----------
#Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Training & Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid()
plt.show()
