"""
Data processing module for handling data.csv with industry-specific filtering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
import config

def get_industries(data_file=None):
    """
    Get a list of all unique industries in the dataset.
    
    Args:
        data_file (str, optional): Path to the data file. If None, uses the default 'data/data.csv'.
        
    Returns:
        list: List of unique industry names
    """
    # Always use data/data.csv if not specified
    if data_file is None:
        data_file = "data/data.csv"
    
    # Load the CSV
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = Path(config.PROJECT_ROOT) / data_path
    
    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get unique industries
    industries = df['industry'].unique().tolist()
    
    return industries

def load_data(data_file=None, industry=None):
    """
    Load and preprocess data from data.csv, optionally filtering by industry.
    
    Args:
        data_file (str, optional): Path to the data file. If None, uses the default 'data/data.csv'.
        industry (str, optional): If provided, only load data for this specific industry.
        
    Returns:
        tuple: (dates, scaled_data, scaler, feature_names)
    """
    # Always use data/data.csv if not specified
    if data_file is None:
        data_file = "data/data.csv"
    
    # Load the CSV
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = Path(config.PROJECT_ROOT) / data_path
    
    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Filter by industry if specified
    if industry is not None:
        df = df[df['industry'] == industry].copy()
        
        if len(df) == 0:
            raise ValueError(f"No data found for industry: {industry}")
    
    # Drop any rows with NaN in the target column
    df = df.dropna(subset=['y'])
    
    # Round float columns to 4 decimal places for numerical stability
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(4)
    
    # Sort by date
    df = df.sort_values('date')
    
    # If single industry, we don't need to one-hot encode
    if industry is not None:
        # Drop industry column since it's all the same value
        df = df.drop(columns=['industry'])
        
        # For the features, we'll use everything except date and the target y
        feature_cols = df.drop(['date', 'y'], axis=1).columns
        
        # Include y in the scaled data for convenience
        all_data = df.drop(['date'], axis=1)
        all_cols = all_data.columns.tolist()
    else:
        # Sort by industry and date
        df = df.sort_values(['industry', 'date'])
        
        # One-hot encode the industry column
        df = pd.get_dummies(df, columns=['industry'], prefix='ind')
        
        # For the features, we'll use everything except date and the target y
        feature_cols = df.drop(['date', 'y'], axis=1).columns
        
        # Include y in the scaled data for convenience
        all_data = df.drop(['date'], axis=1)
        all_cols = all_data.columns.tolist()
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_data)
    
    return df['date'].values, scaled_data, scaler, all_cols

def create_sequences(data, seq_length=None):
    """
    Create sliding window sequences from the data.
    
    Args:
        data (np.ndarray): Scaled data
        seq_length (int, optional): Length of each sequence
        
    Returns:
        tuple: (X, y) where X are input sequences and y are targets
    """
    # Use default sequence length if none specified
    if seq_length is None:
        seq_length = config.TRAINING_CONFIG['sequence_length']
    
    # Create sequences for time series prediction
    n_samples = len(data) - seq_length
    n_features = data.shape[1]
    
    X = np.zeros((n_samples, seq_length, n_features))
    y = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        X[i] = data[i:(i + seq_length)]
        y[i] = data[i + seq_length]
    
    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)
    
    return X, y

def split_data(X, y):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X (torch.Tensor): Input sequences
        y (torch.Tensor): Target values
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=config.TRAINING_CONFIG['test_size'],
        shuffle=False  # Keep time series order
    )
    
    # Then split the rest into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config.TRAINING_CONFIG['val_size'],
        shuffle=False  # Keep time series order
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_scaler(scaler, feature_names, industry=None):
    """
    Save the scaler and feature names.
    
    Args:
        scaler (StandardScaler): The fitted scaler
        feature_names (list): Feature names
        industry (str, optional): Industry name if using industry-specific models
    """
    # Save the scaler so we can use it later
    import joblib
    models_dir = Path(config.DATA_CONFIG['models_dir'])
    
    # If industry-specific, save in industry-specific folder
    if industry is not None:
        models_dir = models_dir / industry
        
    models_dir.mkdir(exist_ok=True, parents=True)
    
    scaler_path = models_dir / 'scaler.joblib'
    joblib.dump((scaler, feature_names), scaler_path)
    print(f"Scaler saved to {scaler_path}")

def load_scaler(industry=None):
    """
    Load the saved scaler and feature names.
    
    Args:
        industry (str, optional): Industry name if using industry-specific models
        
    Returns:
        tuple: (scaler, feature_names)
    """
    # Load the saved scaler
    import joblib
    models_dir = Path(config.DATA_CONFIG['models_dir'])
    
    # If industry-specific, load from industry-specific folder
    if industry is not None:
        models_dir = models_dir / industry
    
    scaler_path = models_dir / 'scaler.joblib'
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Could not find scaler at {scaler_path}. Have you trained the model yet?")
    
    return joblib.load(scaler_path) 