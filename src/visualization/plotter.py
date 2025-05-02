import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_loss_curves(train_losses, val_losses, savepath=None, title=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        savepath (str, optional): Path to save the plot
        title (str, optional): Custom title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    
    if title:
        plt.title(title)
        
    plt.legend()
    plt.grid(True)
    
    if savepath:
        # Ensure directory exists
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_predictions(dates, y_true, y_pred, feature_names, n_plots=3, savepath=None, title=None):
    """
    Plot actual vs. predicted values for selected features.
    
    Args:
        dates (array): Dates for x-axis
        y_true (array): True values
        y_pred (array): Predicted values
        feature_names (list): Names of features
        n_plots (int, optional): Number of features to plot
        savepath (str, optional): Path to save the plot
        title (str, optional): Custom title for the plot
    """
    idxs = np.arange(min(n_plots, y_true.shape[1]))
    fig, axes = plt.subplots(len(idxs), 1, figsize=(12, 4*len(idxs)))
    
    if len(idxs)==1:
        axes=[axes]
        
    for ax, i in zip(axes, idxs):
        ax.plot(dates, y_true[:,i], label=f"Actual {feature_names[i]}")
        ax.plot(dates, y_pred[:,i], "--", label=f"Pred {feature_names[i]}")
        ax.legend()
        ax.set_ylabel(feature_names[i])
        ax.grid(True, alpha=0.3)
        
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.95)
    
    plt.xlabel("Time")
    plt.tight_layout()
    
    if savepath:
        # Ensure directory exists
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def print_metrics(metrics: dict):
    """
    Pretty-print the metrics dict:
      { feature: {'MSE':…, 'MAE':…, 'MAPE':…}, … }
      
    Args:
        metrics (dict): Dictionary of metrics for each feature
    """
    print("\nFeature-wise metrics:")
    print("-" * 60)
    print(f"{'Feature':<25} {'MSE':<12} {'MAE':<12} {'MAPE':<12}")
    print("-" * 60)
    
    for feat, vals in metrics.items():
        print(f"{feat:<25} {vals['MSE']:<12.4f} {vals['MAE']:<12.4f} {vals['MAPE']:<12.2f}%")
    print("-" * 60)

def plot_industry_comparison(industry_metrics, metric='MSE', top_n=10, savepath=None):
    """
    Plot a comparison of a specific metric across industries.
    
    Args:
        industry_metrics (dict): Dictionary mapping industry names to metric dictionaries
        metric (str): Which metric to compare ('MSE', 'MAE', or 'MAPE')
        top_n (int): How many industries to show (sorted by metric)
        savepath (str, optional): Path to save the plot
    """
    # Extract the metric values for each industry
    industries = []
    values = []
    
    for industry, metrics in industry_metrics.items():
        if 'y' in metrics:  # Only include if we have metrics for the target feature
            industries.append(industry)
            values.append(metrics['y'][metric])
    
    # Sort by metric value (ascending)
    sorted_indices = np.argsort(values)
    industries = [industries[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Limit to top_n
    if top_n > 0 and len(industries) > top_n:
        industries = industries[:top_n]
        values = values[:top_n]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.barh(industries, values)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(value + max(values) * 0.01, 
                 bar.get_y() + bar.get_height()/2, 
                 f'{value:.4f}', 
                 va='center')
    
    plt.title(f"Industry Comparison - {metric} for target 'y'")
    plt.xlabel(metric)
    plt.tight_layout()
    
    if savepath:
        # Ensure directory exists
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
