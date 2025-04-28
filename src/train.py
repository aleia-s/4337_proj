import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import joblib

# config & hyper-params
from src.config import TRAINING_CONFIG, DATA_CONFIG
# data prep
from src.data_processing import load_data, create_sequences, split_data, save_scalers
# model
from src.LSTNet import LSTNet
# visualization helpers
from src.visualization.plotter import plot_loss_curves, plot_predictions, print_metrics

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_names: list):
    metrics = {}
    for i, feat in enumerate(feature_names):
        mse  = np.mean((y_true[:, i] - y_pred[:, i])**2)
        mae  = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        mape = 100.0 * np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i]))
        metrics[feat] = dict(MSE=mse, MAE=mae, MAPE=mape)
    return metrics

def train_model(model, X_train, y_train, X_val, y_val):
    cfg = TRAINING_CONFIG
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    train_losses, val_losses = [], []

    for epoch in range(1, cfg['epochs']+1):
        # -- train --
        model.train()
        total_loss = 0.0
        for i in range(0, len(X_train), cfg['batch_size']):
            bx = X_train[i:i+cfg['batch_size']]
            by = y_train[i:i+cfg['batch_size']]
            optimizer.zero_grad()
            preds = model(bx)
            loss  = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # -- validate --
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss  = criterion(val_preds, y_val).item()

        train_losses.append(total_loss / len(X_train))
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch==1:
            print(f"Epoch {epoch}/{cfg['epochs']}  "
                  f"Train {train_losses[-1]:.4f}  Val {val_losses[-1]:.4f}")

    return train_losses, val_losses

def save_model(model: nn.Module, feature_names: list):
    out_dir = Path(DATA_CONFIG['models_dir'])
    out_dir.mkdir(exist_ok=True, parents=True)
    path = out_dir / 'lstnet_model.pth'
    torch.save({
        'state_dict': model.state_dict(),
        'feature_names': feature_names
    }, path)
    print(f"Model → {path}")

def main():
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # 1) load & preprocess
    print("\n1) Loading data...")
    dates, X_raw, y_raw, feat_names = load_data()
    print("Features:", feat_names)

    # 2) split raw data into train/validation/test partitions
    print("\n2) Splitting raw data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_raw, y_raw)

    # 3) fit scalers on train only
    print("\n3) Scaling data and persisting scalers")
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    # transform each train/test/val split
    X_train = scaler_X.transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # persist scalers to disk
    import joblib
    models_dir = Path(DATA_CONFIG['models_dir'])
    models_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump((scaler_X, scaler_y, feat_names), 
                 models_dir / 'scalers_and_features.joblib')

    # 4) build 4-week sliding window sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length=4)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length=4)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length=4)

    print("\n4) Init model…")
    model = LSTNet(num_features=len(feat_names), device=device)

    # 5) train
    print("\n5) Training…")
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val)

    # plot losses
    plot_loss_curves(train_losses, val_losses)

    # 6) save model
    print("\n6) Saving model…")
    save_model(model, feat_names)

    # 7) final eval
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
        truth = y_test.cpu().numpy()

    metrics = calculate_metrics(truth, preds, feat_names)
    print_metrics(metrics)

    # optional: plot_predictions(dates_test, truth, preds, feat_names)

if __name__=="__main__":
    main()
