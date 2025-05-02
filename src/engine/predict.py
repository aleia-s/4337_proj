"""
Script for making predictions with trained LSTNet models.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import config
from src.data.data_processor import load_data, create_sequences, get_industries, load_scaler
from src.visualization.plotter import plot_predictions, print_metrics, plot_industry_comparison
from OLD_FILES.LSTNet import LSTNet
import argparse
import time

def load_model(industry=None):
    """
    Load a trained model.
    
    Args:
        industry (str, optional): Industry name for industry-specific models
        
    Returns:
        tuple: (model, feature_names)
    """
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model
    models_dir = Path(config.DATA_CONFIG['models_dir'])
    
    # If industry-specific, load from industry-specific folder
    if industry is not None:
        models_dir = models_dir / industry
    
    model_path = models_dir / 'lstnet_model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model at {model_path}. Have you trained the model yet?")
    
    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    feature_names = checkpoint['feature_names']
    
    # Initialize the model
    model = LSTNet(
        num_features=len(feature_names),
        device=device
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, feature_names

def make_predictions(model, X):
    """
    Make predictions using a trained model.
    
    Args:
        model (LSTNet): The trained model
        X (torch.Tensor): Input sequences
        
    Returns:
        np.ndarray: Predicted values
    """
    with torch.no_grad():
        y_pred = model(X).cpu().numpy()
    return y_pred

def predict_for_industry(industry=None):
    """
    Make predictions for a specific industry or the overall model.
    
    Args:
        industry (str, optional): Industry name for industry-specific predictions
        
    Returns:
        tuple: (predictions, metrics)
    """
    # Load the model and feature names
    model, feature_names = load_model(industry)
    
    # Load the scaler
    scaler, scaler_feature_names = load_scaler(industry)
    
    # Load and preprocess the data
    dates, scaled_data, _, _ = load_data(industry=industry)
    
    # Create sequences
    X, y_true = create_sequences(scaled_data)
    
    # Make predictions
    y_pred = make_predictions(model, X)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true.cpu().numpy(), y_pred, feature_names)
    
    return dates, y_true.cpu().numpy(), y_pred, feature_names, metrics

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

def predict_multiple_industries(top_n=5, metric='MSE'):
    """
    Make predictions for multiple industries and compare them.
    
    Args:
        top_n (int): Number of top industries to predict
        metric (str): Metric to use for ranking ('MSE', 'MAE', or 'MAPE')
    """
    # Get all industries
    industries = get_industries()
    print(f"Found {len(industries)} industries")
    
    # Dictionary to store metrics for each industry
    industry_metrics = {}
    
    # Predict for each industry
    for i, industry in enumerate(industries):
        print(f"\nIndustry {i+1}/{len(industries)}: {industry}")
        
        try:
            models_dir = Path(config.DATA_CONFIG['models_dir']) / industry
            model_path = models_dir / 'lstnet_model.pth'
            
            # Skip if model doesn't exist
            if not model_path.exists():
                print(f"No model found for {industry}, skipping...")
                continue
            
            # Make predictions
            dates, y_true, y_pred, feature_names, metrics = predict_for_industry(industry)
            
            # Store y metrics for comparison
            if 'y' in metrics:
                industry_metrics[industry] = metrics['y']
                
                # Print metrics for this industry
                print(f"MSE: {metrics['y']['MSE']:.4f}, MAE: {metrics['y']['MAE']:.4f}, MAPE: {metrics['y']['MAPE']:.2f}%")
                
        except Exception as e:
            print(f"Error making predictions for industry {industry}: {str(e)}")
    
    # Sort industries by the specified metric
    sorted_industries = sorted(
        industry_metrics.keys(),
        key=lambda ind: industry_metrics[ind][metric]
    )
    
    # Get top N industries
    top_industries = sorted_industries[:top_n]
    
    print("\n" + "="*70)
    print(f"Top {top_n} Industries by {metric}:")
    print("="*70)
    print(f"{'Industry':<30} {'MSE':<10} {'MAE':<10} {'MAPE':<10}")
    print("-"*70)
    
    for industry in top_industries:
        metrics = industry_metrics[industry]
        print(f"{industry:<30} {metrics['MSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['MAPE']:<10.2f}%")
    
    # Create comparison visualizations directory
    comparisons_dir = Path(config.DATA_CONFIG['visualizations_dir']) / 'industry_comparisons'
    comparisons_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the metrics comparison plot
    plot_industry_comparison(
        industry_metrics, 
        metric=metric, 
        top_n=top_n,
        savepath=comparisons_dir / f'top_{top_n}_industries_by_{metric}.png'
    )
        
    # Make detailed predictions for top industries
    for industry in top_industries:
        print(f"\nGenerating detailed predictions for {industry}...")
        
        # Make predictions
        dates, y_true, y_pred, feature_names, metrics = predict_for_industry(industry)
        
        # Plot predictions for y
        if 'y' in feature_names:
            y_index = feature_names.index('y')
            
            visualizations_dir = Path(config.DATA_CONFIG['visualizations_dir']) / 'predictions' / industry
            visualizations_dir.mkdir(exist_ok=True, parents=True)
            
            plot_predictions(
                dates[-len(y_pred):],
                y_true[:, [y_index]],
                y_pred[:, [y_index]],
                ['y'],
                savepath=visualizations_dir / 'employment_predictions.png',
                title=f"Employment Predictions - {industry}"
            )

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained LSTNet models')
    parser.add_argument('--industry', type=str, help='Industry to predict for', default=None)
    parser.add_argument('--compare', action='store_true', help='Compare predictions across industries')
    parser.add_argument('--top', type=int, help='Number of top industries to compare', default=5)
    parser.add_argument('--metric', type=str, help='Metric for comparison (MSE, MAE, MAPE)', default='MSE')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Ensure results directories exist
    for dir_path in [config.DATA_CONFIG['models_dir'], config.DATA_CONFIG['visualizations_dir']]:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
    
    if args.compare:
        predict_multiple_industries(top_n=args.top, metric=args.metric)
    else:
        # If no specific industry is provided, list available industries
        if args.industry is None:
            industries = get_industries()
            print("Available industries:")
            for i, industry in enumerate(industries):
                print(f"{i+1}. {industry}")
            
            try:
                industry_idx = int(input(f"Enter industry number (1-{len(industries)}) or 0 for all data: ")) - 1
                if industry_idx < 0:
                    args.industry = None  # Use all data
                elif 0 <= industry_idx < len(industries):
                    args.industry = industries[industry_idx]
                else:
                    print("Invalid selection. Using all data.")
                    args.industry = None
            except ValueError:
                print("Invalid input. Using all data.")
                args.industry = None
        
        # Display which industry we're predicting for
        industry_display = args.industry if args.industry else "All Industries"
        print(f"\nMaking predictions for: {industry_display}")
        
        # Make predictions
        dates, y_true, y_pred, feature_names, metrics = predict_for_industry(args.industry)
        
        # Print metrics
        print_metrics(metrics)
        
        # Plot predictions for y
        if 'y' in feature_names:
            y_index = feature_names.index('y')
            
            visualizations_dir = Path(config.DATA_CONFIG['visualizations_dir'])
            if args.industry:
                visualizations_dir = visualizations_dir / args.industry
            visualizations_dir.mkdir(exist_ok=True, parents=True)
            
            plot_predictions(
                dates[-len(y_pred):],
                y_true[:, [y_index]],
                y_pred[:, [y_index]],
                ['y'],
                savepath=visualizations_dir / 'employment_predictions.png',
                title=f"Employment Predictions - {industry_display}"
            )
    
    end_time = time.time()
    print(f"\nTotal prediction time: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main() 