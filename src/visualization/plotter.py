import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config
import matplotlib as mpl

def set_academic_style():
    """
    Apply academic paper style settings with Times New Roman font.
    """
    # Set font family to Times New Roman
    plt.rcParams['font.family'] = config.VISUALIZATION_CONFIG['font_family']
    
    # Set font sizes
    plt.rcParams['font.size'] = config.VISUALIZATION_CONFIG['font_size']
    plt.rcParams['axes.titlesize'] = config.VISUALIZATION_CONFIG['title_size']
    plt.rcParams['axes.labelsize'] = config.VISUALIZATION_CONFIG['label_size']
    plt.rcParams['xtick.labelsize'] = config.VISUALIZATION_CONFIG['tick_size']
    plt.rcParams['ytick.labelsize'] = config.VISUALIZATION_CONFIG['tick_size']
    plt.rcParams['legend.fontsize'] = config.VISUALIZATION_CONFIG['legend_size']
    
    # High-quality output
    plt.rcParams['figure.dpi'] = config.VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.dpi'] = config.VISUALIZATION_CONFIG['dpi']
    
    # Clean, minimalist style
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def plot_loss_curves(train_losses, val_losses, savepath=None, title=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        savepath (str, optional): Path to save the plot
        title (str, optional): Custom title for the plot
    """
    set_academic_style()
    
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figure_size'])
    plt.plot(train_losses, label="Train Loss", 
             color=config.VISUALIZATION_CONFIG['colors']['actual'],
             linewidth=config.VISUALIZATION_CONFIG['line_width'],
             linestyle=config.VISUALIZATION_CONFIG['line_styles']['actual'])
    
    plt.plot(val_losses, label="Val Loss",
             color=config.VISUALIZATION_CONFIG['colors']['pred'],
             linewidth=config.VISUALIZATION_CONFIG['line_width'],
             linestyle=config.VISUALIZATION_CONFIG['line_styles']['pred'])
    
    # For academic papers, we'll keep axis labels minimal or remove them
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    if title and config.VISUALIZATION_CONFIG['show_titles']:
        plt.title(title)
        
    plt.legend(loc=config.VISUALIZATION_CONFIG['legend_loc'])
    plt.grid(True, alpha=config.VISUALIZATION_CONFIG['grid_alpha'], 
             linestyle=config.VISUALIZATION_CONFIG['grid_style'])
    
    if savepath:
        # Ensure directory exists
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches="tight")
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
    set_academic_style()
    
    idxs = np.arange(min(n_plots, y_true.shape[1]))
    fig, axes = plt.subplots(len(idxs), 1, figsize=(12, 4*len(idxs)))
    
    if len(idxs)==1:
        axes=[axes]
        
    for ax, i in zip(axes, idxs):
        ax.plot(dates, y_true[:,i], 
                color=config.VISUALIZATION_CONFIG['colors']['actual'],
                linewidth=config.VISUALIZATION_CONFIG['line_width'],
                linestyle=config.VISUALIZATION_CONFIG['line_styles']['actual'],
                label=f"Actual")
        
        ax.plot(dates, y_pred[:,i], 
                color=config.VISUALIZATION_CONFIG['colors']['pred'],
                linewidth=config.VISUALIZATION_CONFIG['line_width'],
                linestyle=config.VISUALIZATION_CONFIG['line_styles']['pred'],
                label=f"Predicted")
        
        ax.legend(loc=config.VISUALIZATION_CONFIG['legend_loc'])
        
        # For academic papers, minimal or no labels
        # Just showing the feature name as the y-axis label
        ax.set_ylabel(feature_names[i])
        
        # Remove x-axis label for all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xticklabels([])
        
        ax.grid(True, alpha=config.VISUALIZATION_CONFIG['grid_alpha'],
                linestyle=config.VISUALIZATION_CONFIG['grid_style'])
        
    # Add title if provided and if titles are enabled
    if title and config.VISUALIZATION_CONFIG['show_titles']:
        fig.suptitle(title, fontsize=config.VISUALIZATION_CONFIG['title_size'])
        fig.subplots_adjust(top=0.95)
    
    # Only add the time label to the bottom plot
    if len(axes) > 0:
        axes[-1].set_xlabel("Time")
    
    plt.tight_layout()
    
    if savepath:
        # Ensure directory exists
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches="tight")
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
    set_academic_style()
    
    # Extract the metric values for each industry
    industries = []
    values = []
    
    for industry, metrics in industry_metrics.items():
        if 'y' in metrics:  # Only include if we have metrics for the target feature
            industries.append(industry)
            values.append(metrics['y'][metric])
    
    # Sort by metric value (ascending for errors, higher is worse)
    sorted_indices = np.argsort(values)
    
    # If top_n is negative, we want the worst performers (from the end)
    if top_n < 0:
        sorted_indices = sorted_indices[top_n:]  # Last n elements
        top_n = abs(top_n)
    else:
        sorted_indices = sorted_indices[:top_n]  # First n elements
    
    industries = [industries[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Create the plot
    plt.figure(figsize=config.VISUALIZATION_CONFIG['figure_size'])
    
    # Use academic color scheme
    bars = plt.barh(industries, values, 
                   color=config.VISUALIZATION_CONFIG['colors']['mape_bar'],
                   edgecolor=config.VISUALIZATION_CONFIG['colors']['actual'],
                   linewidth=0.5)
    
    # Add value labels if configured
    if config.VISUALIZATION_CONFIG['show_value_labels']:
        decimal_places = config.VISUALIZATION_CONFIG['value_label_decimals'].get(metric.lower(), 3)
        format_str = f'{{:.{decimal_places}f}}'
        
        for bar, value in zip(bars, values):
            plt.text(value + max(values) * config.VISUALIZATION_CONFIG['label_offset'], 
                     bar.get_y() + bar.get_height()/2, 
                     format_str.format(value), 
                     va='center',
                     fontsize=config.VISUALIZATION_CONFIG['tick_size'])
    
    # Minimize labels for academic style
    if config.VISUALIZATION_CONFIG['show_titles']:
        plt.title(f"Industry Comparison - {metric}")
    
    # Very minimal axis labels for academic paper
    plt.xlabel(metric)
    # No y-axis label needed as the industry names are self-explanatory
    
    plt.tight_layout()
    
    if savepath:
        # Ensure directory exists
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches="tight")
    else:
        plt.show()
    plt.close()
