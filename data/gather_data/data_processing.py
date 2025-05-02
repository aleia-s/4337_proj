"""
data_processor.py

Handles both formats:
1. multivariate_unemployment_LSTNet.csv: Time series data with multiple unemployment metrics
2. all_industries.csv: Multiple industries with various metrics per industry
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG, TRAINING_CONFIG, ACTIVE_DATASET

def load_data(data_file=None):
    if data_file is None:
        data_file = DATA_CONFIG['default_data_file']
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = Path(DATA_CONFIG['data_dir']).parent / data_path

    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Determine which dataset format we're working with
    if ACTIVE_DATASET == "unemployment":
        # Format 1: Unemployment data with multiple features per row (no industry column)
        # Keep all columns as features except date
        X = df.drop(['date'], axis=1).values
        # Target is all the same features (for forecasting all columns)
        y = X
        feature_names = df.drop(['date'], axis=1).columns.tolist()
    else:
        # Format 2: Industries data with multiple industries
        df = df.dropna(subset=['y'])
        float_cols = df.select_dtypes(include=['float64']).columns 
        df[float_cols] = df[float_cols].round(4)
        
        df = df.sort_values(['industry', 'date'])
        df = pd.get_dummies(df, columns=['industry'], prefix='ind')

        y = df['y'].values.reshape(-1,1)                # target
        X = df.drop(['date','y'], axis=1).values        # all other cols
        feature_names = df.drop(['date','y'], axis=1).columns.tolist()

    return df['date'].values, X, y, feature_names

def create_sequences(X, y, seq_length=None):
    # default seq_length
    if seq_length is None:
        seq_length = TRAINING_CONFIG['sequence_length']

    n_samples = len(X) - seq_length
    n_feats_x = X.shape[1]
    
    # Determine shape for y_seq based on y's dimensionality
    if len(y.shape) > 1 and y.shape[1] > 1:
        # For unemployment dataset, y has same shape as X (multi-output)
        n_feats_y = y.shape[1]
        y_seq = np.zeros((n_samples, n_feats_y), dtype=np.float32)
    else:
        # For industries dataset, y is a single column
        y_seq = np.zeros((n_samples, 1), dtype=np.float32)
    
    # allocate
    X_seq = np.zeros((n_samples, seq_length, n_feats_x), dtype=np.float32)

    for i in range(n_samples):
        X_seq[i] = X[i : i + seq_length]
        if len(y.shape) > 1 and y.shape[1] > 1:
            # For multi-output case (unemployment dataset)
            y_seq[i] = y[i + seq_length]
        else:
            # For single-output case (industries dataset)
            y_seq[i, 0] = y[i + seq_length]

    # to torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.from_numpy(X_seq).to(device)
    y_t = torch.from_numpy(y_seq).to(device)

    return X_t, y_t

def split_data(X, y):
    # first test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TRAINING_CONFIG['test_size'], shuffle=False
    )
    # then train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=TRAINING_CONFIG['val_size'], shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

