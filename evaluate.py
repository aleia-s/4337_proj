"""
Script for loading and evaluating a trained LSTNet model on the all_industries.csv data.
"""
import torch
import numpy as np
from pathlib import Path
import config
from src.data.data_processor import load_data, create_sequences, load_scaler
from src.visualization.plotter import (
    plot_predictions, 
    print_metrics,
)
from OLD_FILES.LSTNet import LSTNet

def load_trained_model():
    """
    Load the trained model and scaler.
    
    Returns:
        tuple: (model, scaler, feature_names)
    """
    # Load model
    model_path = Path(config.DATA_CONFIG['models_dir']) / 'lstnet_model.pth'
    checkpoint = torch.load(model_path)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTNet(
        num_features=len(checkpoint['feature_names']),
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    scaler, feature_names = load_scaler()
    
    return model, scaler, feature_names

def calculate_metrics(y_true, y_pred, feature_names):
    """
    Calculate metrics for each feature.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        feature_names (list): List of feature names
        
    Returns:
        dict: Dictionary of metrics
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

def evaluate_model(model, scaler, feature_names, data_file=None):
    """
    Evaluate the model on the specified data file.
    
    Args:
        model (LSTNet): The trained model
        scaler (StandardScaler): The fitted scaler
        feature_names (list): List of feature names
        data_file (str, optional): Path to the data file to evaluate on
    """
    # Load and preprocess data
    print("\nLoading data...")
    dates, scaled_data, _, _ = load_data(data_file)
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(scaled_data)
    
    # Generate predictions
    print("Generating predictions...")
    with torch.no_grad():
        predictions = model(X)
    
    # Convert to numpy arrays
    predictions = predictions.cpu().numpy()
    y = y.cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(y, predictions, feature_names)
    
    # Print overall metrics
    print("\nEvaluation Results:")
    print_metrics(metrics)
    
    # Focus on the target feature 'y'
    if 'y' in feature_names:
        y_index = feature_names.index('y')
        print(f"\nPerformance on target feature 'y':")
        print(f"MSE: {metrics['y']['MSE']:.4f}")
        print(f"MAE: {metrics['y']['MAE']:.4f}")
        print(f"MAPE: {metrics['y']['MAPE']:.2f}%")
    
    # Try to visualize results for selected features
    try:
        # Get visualizations directory or use 'results' as fallback
        if 'visualizations_dir' in config.DATA_CONFIG:
            visualizations_dir = Path(config.DATA_CONFIG['visualizations_dir'])
        else:
            visualizations_dir = Path('results/visualizations')
            
        visualizations_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot time series for up to 3 features, including 'y' if available
        features_to_plot = []
        if 'y' in feature_names:
            features_to_plot.append(feature_names.index('y'))
        
        # Add industry features (up to 2 more)
        ind_features = [i for i, f in enumerate(feature_names) if f.startswith('ind_')]
        features_to_plot.extend(ind_features[:2])
        
        # Limit to max 3 features
        features_to_plot = features_to_plot[:3]
        
        # Plot predictions vs actual
        print("\nPlotting predictions...")
        selected_features = [feature_names[i] for i in features_to_plot]
        y_selected = y[:, features_to_plot]
        pred_selected = predictions[:, features_to_plot]
        
        plot_predictions(
            dates[-len(predictions):], 
            y_selected, 
            pred_selected, 
            selected_features,
            savepath=visualizations_dir / 'predictions.png'
        )
        
        print(f"\nEvaluation complete! Results have been saved to the '{visualizations_dir}' directory.")
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        print("Evaluation complete without visualizations.")

def main():
    # Load trained model and scaler
    print("Loading trained model...")
    model, scaler, feature_names = load_trained_model()
    
    # Evaluate model
    evaluate_model(model, scaler, feature_names)

if __name__ == "__main__":
    main() 