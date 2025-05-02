import torch
import numpy as np
import pandas as pd
from pathlib import Path
import config
from src.data.data_processor import load_data, create_sequences, get_industries, load_scaler
from src.visualization.plotter import plot_predictions, plot_loss_curves, print_metrics, plot_industry_comparison, plot_model_comparison_metrics, plot_all_industries_mape
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

def evaluate_industry_model(industry=None):
    """
    Evaluate a trained model for a specific industry and generate visualizations.
    
    Args:
        industry (str, optional): Industry name for industry-specific evaluation
        
    Returns:
        tuple: (dates, y_true, y_pred, feature_names, metrics)
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
    model.eval()
    with torch.no_grad():
        y_pred = model(X).cpu().numpy()
        y_true = y_true.cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, feature_names)
    
    # Generate visualizations
    visualizations_dir = Path(config.DATA_CONFIG['visualizations_dir'])
    if industry:
        visualizations_dir = visualizations_dir / industry
    visualizations_dir.mkdir(exist_ok=True, parents=True)
    
    # If 'y' is in the features, create employment prediction visualization
    if 'y' in feature_names:
        y_index = feature_names.index('y')
        
        # Display metrics
        print(f"\nPerformance on employment ('y'):")
        print(f"MSE: {metrics['y']['MSE']:.4f}")
        print(f"MAE: {metrics['y']['MAE']:.4f}")
        print(f"SMAPE: {metrics['y']['SMAPE']:.2f}%")
        
        # Plot actual vs predicted for employment
        industry_display = industry if industry else "All Industries"
        plot_predictions(
            dates[-len(y_pred):], 
            y_true[:, [y_index]], 
            y_pred[:, [y_index]], 
            ['y'],
            savepath=visualizations_dir / 'employment_predictions.png',
            title=f"Employment Predictions - {industry_display}"
        )
    
    return dates, y_true, y_pred, feature_names, metrics

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
        
        # Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
        # SMAPE = 100/n * sum(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
        # This avoids division by zero issues and is symmetric
        denominator = np.abs(y_true[:, i]) + np.abs(y_pred[:, i])
        # Avoid division by zero (where both true and predicted are zero)
        mask = denominator != 0
        smape = 100.0 * np.mean(2.0 * np.abs(y_true[:, i][mask] - y_pred[:, i][mask]) / denominator[mask])
        
        metrics[feature] = {
            'MSE': mse,
            'MAE': mae,
            'SMAPE': smape
        }
    
    return metrics

def evaluate_all_industries():
    """
    Evaluate models for all industries and generate comparison visualizations.
    """
    # Get all industries
    industries = get_industries()
    print(f"Found {len(industries)} industries")
    
    # Dictionary to store metrics for each industry
    industry_metrics = {}
    
    # Dictionary to store feature metrics for model comparison chart
    model_comparison_data = {}
    
    # Evaluate each industry
    for i, industry in enumerate(industries):
        print(f"\nIndustry {i+1}/{len(industries)}: {industry}")
        
        try:
            models_dir = Path(config.DATA_CONFIG['models_dir']) / industry
            model_path = models_dir / 'lstnet_model.pth'
            
            # Skip if model doesn't exist
            if not model_path.exists():
                print(f"No model found for {industry}, skipping...")
                continue
            
            # Evaluate model
            _, _, _, feature_names, metrics = evaluate_industry_model(industry)
            
            # Store metrics for comparison
            if 'y' in metrics:
                industry_metrics[industry] = metrics['y']
                model_comparison_data[industry] = metrics
                
        except Exception as e:
            print(f"Error evaluating model for industry {industry}: {str(e)}")
    
    # Create comparison visualizations directory
    visualizations_dir = Path(config.DATA_CONFIG['visualizations_dir'])
    comparisons_dir = visualizations_dir / 'industry_comparisons'
    comparisons_dir.mkdir(exist_ok=True, parents=True)
    
    # Print summary of industry performances
    print("\n" + "="*70)
    print("Summary of Industry Model Performance (employment):")
    print("="*70)
    print(f"{'Industry':<30} {'MSE':<10} {'MAE':<10} {'SMAPE':<12}")
    print("-"*70)
    
    for industry, metrics in industry_metrics.items():
        print(f"{industry:<30} {metrics['MSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['SMAPE']:<12.2f}%")
    
    # Create a single chart showing SMAPE values for all industries
    if len(model_comparison_data) > 0:
        # Chart 1: All industries
        plot_all_industries_mape(
            model_comparison_data,
            feature_name='y',
            metric_name='SMAPE',
            savepath=comparisons_dir / 'all_industries_smape_comparison.png',
            title="Industry Model Performance Comparison (SMAPE %)"
        )
        
        # Chart 2: Only industries with SMAPE < 30%
        # Create a filtered copy of the metrics dictionary
        filtered_metrics = {}
        for industry, metrics in model_comparison_data.items():
            if metrics['y']['SMAPE'] < 40.0:  # Only include industries with SMAPE < 30%
                filtered_metrics[industry] = metrics
        
        if len(filtered_metrics) > 0:
            plot_all_industries_mape(
                filtered_metrics,
                feature_name='y',
                metric_name='SMAPE',
                savepath=comparisons_dir / 'filtered_industries_smape_comparison.png',
                title="Industry Model Performance Comparison"
            )
            print(f"\nCreated filtered chart with {len(filtered_metrics)} industries (SMAPE < 30%)")
        else:
            print("\nNo industries with SMAPE < 30% found for filtered chart")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models and generate visualizations')
    parser.add_argument('--industry', type=str, help='Industry to evaluate', default=None)
    parser.add_argument('--all', action='store_true', help='Evaluate all industries and create comparison visualizations')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Ensure visualization directory exists
    visualizations_dir = Path(config.DATA_CONFIG['visualizations_dir'])
    visualizations_dir.mkdir(exist_ok=True, parents=True)
    
    if args.all:
        evaluate_all_industries()
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
        
        # Display which industry we're evaluating
        industry_display = args.industry if args.industry else "All Industries"
        print(f"\nEvaluating model for: {industry_display}")
        
        # Evaluate model
        _, _, _, feature_names, metrics = evaluate_industry_model(args.industry)
        
        # Print metrics for all features
        print_metrics(metrics)
    
    end_time = time.time()
    print(f"\nTotal evaluation time: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main() 