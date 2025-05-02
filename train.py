"""
Main script for training the LSTNet model on the all_industries.csv data.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import config
from src.data.data_processor import load_data, create_sequences, split_data, save_scaler
from src.visualization.plotter import plot_predictions, plot_loss_curves, print_metrics
from OLD_FILES.LSTNet import LSTNet

def calculate_metrics(y_true, y_pred, feature_names):
    """
    Calculate various metrics for each feature.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        feature_names (list): List of feature names
        
    Returns:
        dict: Dictionary of metrics for each feature
    """
    metrics = {}
    for i, feature in enumerate(feature_names):
        mse = np.mean((y_true[:, i] - y_pred[:, i])**2)
        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
        
        metrics[feature] = {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape
        }
    
    return metrics

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the LSTNet model.
    
    Args:
        model (LSTNet): The model to train
        X_train (torch.Tensor): Training input sequences
        y_train (torch.Tensor): Training target sequences
        X_val (torch.Tensor): Validation input sequences
        y_val (torch.Tensor): Validation target sequences
        
    Returns:
        tuple: (train_losses, val_losses)
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING_CONFIG['learning_rate'])
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config.TRAINING_CONFIG['epochs']):
        # Training phase
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), config.TRAINING_CONFIG['batch_size']):
            batch_X = X_train[i:i+config.TRAINING_CONFIG['batch_size']]
            batch_y = y_train[i:i+config.TRAINING_CONFIG['batch_size']]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        # Record losses
        train_losses.append(total_loss / len(X_train))
        val_losses.append(val_loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config.TRAINING_CONFIG["epochs"]}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def save_model(model, feature_names):
    """
    Save the trained model.
    
    Args:
        model (LSTNet): The trained model
        feature_names (list): List of feature names
    """
    models_dir = Path(config.DATA_CONFIG['models_dir'])
    models_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = models_dir / 'lstnet_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names
    }, model_path)
    print(f"Model saved to {model_path}")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    dates, scaled_data, scaler, feature_names = load_data()
    print(f"Dataset shape: {scaled_data.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Step 2: Create sequences
    print("\nStep 2: Creating sequences...")
    X, y = create_sequences(scaled_data)
    print(f"Sequence shape X: {X.shape}, y: {y.shape}")
    
    # Step 3: Split data
    print("\nStep 3: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Step 4: Initialize model
    print("\nStep 4: Initializing model...")
    # Set the num_features dynamically
    config.MODEL_CONFIG['num_features'] = scaled_data.shape[1]
    model = LSTNet(
        num_features=scaled_data.shape[1],
        device=device
    )
    
    # Step 5: Train model
    print("\nStep 5: Training model...")
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val)
    
    # Step 6: Plot loss curves
    print("\nStep 6: Plotting loss curves...")
    plot_loss_curves(train_losses, val_losses)
    
    # Step 7: Save model and scaler
    print("\nStep 7: Saving model and scaler...")
    save_model(model, feature_names)
    save_scaler(scaler, feature_names)
    
    # Step 8: Evaluate on test set
    print("\nStep 8: Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        y_true = y_test.cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, feature_names)
    print_metrics(metrics)
    
    # Optional: Plot predictions for specific features
    # You can display predictions for specific features of interest
    # For example, let's focus on the 'y' target feature
    y_index = feature_names.index('y')  # Find the index of 'y' in features
    print(f"\nPerformance on target feature 'y':")
    print(f"MSE: {metrics['y']['MSE']:.4f}")
    print(f"MAE: {metrics['y']['MAE']:.4f}")
    print(f"MAPE: {metrics['y']['MAPE']:.2f}%")
    
    print("\nTraining complete! Model has been saved to models/")

if __name__ == "__main__":
    main() 