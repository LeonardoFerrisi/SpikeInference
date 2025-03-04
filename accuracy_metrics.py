import tensorflow as tf
import numpy as np
# Custom metrics for regression model evaluation
def r_squared(y_true, y_pred):
    """
    Calculates R-squared (coefficient of determination) more robustly in TensorFlow.
    Handles batch-wise calculations correctly.
    """
    # Calculate residual sum of squares
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    
    # Calculate total sum of squares
    # Calculate mean per batch to avoid issues during training
    y_mean = tf.reduce_mean(y_true, axis=0, keepdims=True)
    SS_tot = tf.reduce_sum(tf.square(y_true - y_mean))
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Return R-squared
    return tf.maximum(-1.0, 1 - SS_res / (SS_tot + epsilon))

def mean_absolute_error(y_true, y_pred):
    """
    Calculate the mean absolute error between predictions and ground truth.
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def threshold_accuracy(y_true, y_pred, threshold=0.1):
    """
    Calculate the percentage of predictions within a threshold of the true values.
    This can be used after model predictions are made, not as a Keras metric.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Maximum allowed deviation (as a fraction of the true value)
    
    Returns:
        Percentage of predictions within threshold
    """
    abs_diff = np.abs(y_true - y_pred)
    abs_threshold = np.abs(y_true * threshold)
    within_threshold = np.sum(abs_diff <= abs_threshold)
    return within_threshold / len(y_true) * 100