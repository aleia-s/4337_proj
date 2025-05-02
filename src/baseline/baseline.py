"""
Baseline forecasting models for employment prediction.
This module implements simple baseline models used for comparison in the academic paper.
"""
import numpy as np
import torch

def persistence_forecast(X):
    """
    Persistence forecasting model: predict that the next value will be the same as the current value.
    
    The persistence model is a very simple baseline that assumes the future value
    will be the same as the most recent observation.
    
    Args:
        X (np.ndarray): Input sequences of shape (n_samples, seq_len, n_features)
        
    Returns:
        np.ndarray: Predictions with shape (n_samples, n_features)
    """
    # For each sequence, use the last value as the prediction
    return X[:, -1, :]

def oracle_mean_forecast(y_true):
    """
    Oracle mean forecast: use the mean of all test values as the prediction.
    
    This baseline represents what would be achievable if we knew the mean of the test set
    in advance. It's an "oracle" model because it uses information that wouldn't be 
    available in a real forecasting scenario.
    
    Args:
        y_true (np.ndarray): True values of shape (n_samples, n_features)
        
    Returns:
        np.ndarray: Predictions with the same shape as y_true
    """
    # Calculate the mean along the samples dimension
    mean_values = np.mean(y_true, axis=0, keepdims=True)
    
    # Repeat the mean values for each sample
    return np.repeat(mean_values, y_true.shape[0], axis=0) 