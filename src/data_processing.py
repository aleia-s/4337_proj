"""
data_processor.py

Reworked to handle the merged multi‐industry CSV:
  - one‐hot encodes `industry`
  - separates X (all metrics + industry dummies) and y (“y” column)
  - scales X and y separately
  - builds sliding windows: seq_length → predict next‐step y
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import DATA_CONFIG, TRAINING_CONFIG

def load_data(data_file=None):
    if data_file is None:
        data_file = DATA_CONFIG['default_data_file']
    data_path = Path(DATA_CONFIG['data_dir']) / data_file

    df = pd.read_csv(data_path, parse_dates=['date'])
    if 'y' in df.columns: 
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
    n_feats   = X.shape[1]

    # allocate
    X_seq = np.zeros((n_samples, seq_length, n_feats), dtype=np.float32)
    y_seq = np.zeros((n_samples, 1), dtype=np.float32)

    for i in range(n_samples):
        X_seq[i] = X[i : i + seq_length]
        y_seq[i] = y[i + seq_length]

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

