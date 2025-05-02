"""
Script for evaluating baseline models (persistence and oracle) for industry-level employment forecasting.
This script computes metrics for baseline models to be included in academic paper comparisons.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import time
import argparse
import json

# Add the project root to path to be able to import from config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

from src.data.data_processor import load_data, create_sequences, get_industries
from src.baseline.baseline import persistence_forecast, oracle_mean_forecast

# Helper function to convert numpy values to Python types for JSON serialization
def convert_for_json(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(i) for i in obj]
    else:
        return obj

def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for model evaluation.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        dict: Dictionary of metrics (MSE, RMSE, MAE, SMAPE)
    """
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate SMAPE
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    smape = 100.0 * np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'SMAPE': smape
    }

def evaluate_baseline_for_industry(industry=None):
    """
    Evaluate baseline models for a specific industry.
    
    Args:
        industry (str, optional): Industry name for industry-specific evaluation
        
    Returns:
        dict: Dictionary of metrics for baseline models
    """
    print(f"Evaluating baselines for industry: {industry or 'All Industries'}")
    
    # Load and preprocess the data
    dates, scaled_data, _, feature_names = load_data(industry=industry)
    
    # Create sequences
    X, y_true = create_sequences(scaled_data)
    
    # Convert to numpy if they are torch tensors
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    
    # Get baseline predictions
    persistence_pred = persistence_forecast(X)
    oracle_pred = oracle_mean_forecast(y_true)
    
    results = {}
    
    # For each feature, calculate metrics
    for i, feature in enumerate(feature_names):
        # Persistence model metrics
        persistence_metrics = calculate_metrics(y_true[:, i], persistence_pred[:, i])
        
        # Oracle model metrics
        oracle_metrics = calculate_metrics(y_true[:, i], oracle_pred[:, i])
        
        results[feature] = {
            'persistence': persistence_metrics,
            'oracle': oracle_metrics
        }
        
        # Print metrics for 'y' feature (employment)
        if feature == 'y':
            print(f"\nPerformance on employment ('y'):")
            print(f"Persistence - RMSE: {persistence_metrics['RMSE']:.4f}, MAE: {persistence_metrics['MAE']:.4f}, SMAPE: {persistence_metrics['SMAPE']:.2f}%")
            print(f"Oracle     - RMSE: {oracle_metrics['RMSE']:.4f}, MAE: {oracle_metrics['MAE']:.4f}, SMAPE: {oracle_metrics['SMAPE']:.2f}%")
    
    return results

def evaluate_all_industries():
    """
    Evaluate baseline models for all industries.
    """
    # Get all industries
    industries = get_industries()
    print(f"Found {len(industries)} industries")
    
    # Dictionary to store results for all industries
    all_results = {}
    
    # Evaluate each industry
    for i, industry in enumerate(industries):
        print(f"\nIndustry {i+1}/{len(industries)}: {industry}")
        
        try:
            # Evaluate baselines
            industry_results = evaluate_baseline_for_industry(industry)
            
            # Store results if we have the 'y' feature
            if 'y' in industry_results:
                all_results[industry] = industry_results['y']
                
        except Exception as e:
            print(f"Error evaluating baselines for industry {industry}: {str(e)}")
    
    # Create output directory
    results_dir = Path(__file__).parent / 'baseline_results'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy values to Python types for JSON serialization
    all_results_json = convert_for_json(all_results)
    
    # Save results to JSON
    with open(results_dir / 'baseline_metrics.json', 'w') as f:
        json.dump(all_results_json, f, indent=2)
    
    # Create summary CSV for easy import to paper
    summary_data = []
    
    for industry, metrics in all_results.items():
        summary_data.append({
            'Industry': industry,
            'Persistence_RMSE': float(metrics['persistence']['RMSE']),
            'Persistence_MAE': float(metrics['persistence']['MAE']),
            'Persistence_SMAPE': float(metrics['persistence']['SMAPE']),
            'Oracle_RMSE': float(metrics['oracle']['RMSE']),
            'Oracle_MAE': float(metrics['oracle']['MAE']),
            'Oracle_SMAPE': float(metrics['oracle']['SMAPE'])
        })
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / 'baseline_summary.csv', index=False)
    
    # Create LaTeX table for the academic paper
    create_latex_table(summary_df, results_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("Summary of Baseline Performance (employment, SMAPE %):")
    print("="*70)
    print(f"{'Industry':<30} {'Persistence':<12} {'Oracle':<12}")
    print("-"*70)
    
    for industry, metrics in all_results.items():
        print(f"{industry:<30} {float(metrics['persistence']['SMAPE']):<12.2f} {float(metrics['oracle']['SMAPE']):<12.2f}")
    
    return all_results

def create_latex_table(df, output_dir):
    """
    Create a LaTeX table from the summary DataFrame.
    
    Args:
        df (pd.DataFrame): Summary DataFrame
        output_dir (Path): Directory to save the LaTeX table
    """
    # Create LaTeX table with selected metrics (RMSE and SMAPE)
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Baseline Model Performance by Industry}
\\label{tab:baseline_performance}
\\begin{tabular}{lcccc}
\\toprule
& \\multicolumn{2}{c}{Persistence Model} & \\multicolumn{2}{c}{Oracle Model} \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
Industry & RMSE & SMAPE (\\%) & RMSE & SMAPE (\\%) \\\\
\\midrule
"""
    
    # Add rows for each industry
    for _, row in df.iterrows():
        industry = row['Industry'].replace('&', '\\&')  # Escape & for LaTeX
        latex_table += f"{industry} & {row['Persistence_RMSE']:.4f} & {row['Persistence_SMAPE']:.2f} & {row['Oracle_RMSE']:.4f} & {row['Oracle_SMAPE']:.2f} \\\\\n"
    
    # Add average row
    latex_table += "\\midrule\n"
    latex_table += f"Average & {df['Persistence_RMSE'].mean():.4f} & {df['Persistence_SMAPE'].mean():.2f} & {df['Oracle_RMSE'].mean():.4f} & {df['Oracle_SMAPE'].mean():.2f} \\\\\n"
    
    # Complete the table
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save to file
    with open(output_dir / 'baseline_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to {output_dir / 'baseline_table.tex'}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline models for employment forecasting')
    parser.add_argument('--industry', type=str, help='Industry to evaluate', default=None)
    parser.add_argument('--all', action='store_true', help='Evaluate all industries')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
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
        
        # Evaluate baseline models
        evaluate_baseline_for_industry(args.industry)
    
    end_time = time.time()
    print(f"\nTotal evaluation time: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main() 