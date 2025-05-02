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

def plot_model_comparison_metrics(metrics_dict, feature_name='y', models=None, savepath=None, title="Model Performance Metrics by Feature"):
    """
    Create a bar chart comparing MAPE, MAE, and MSE metrics across different models/industries.
    Similar to the style shown in the example image with grouped bars and patterns.
    
    Args:
        metrics_dict (dict): Dictionary of dictionaries with metrics for each model/industry
                            Format: {model_name: {'feature_name': {'MSE': val, 'MAE': val, 'MAPE': val}}}
        feature_name (str): The feature to plot metrics for (default: 'y')
        models (list): List of model/industry names to include (if None, use all)
        savepath (str): Path to save the figure
        title (str): Title for the plot
    """
    set_academic_style()
    
    # Filter models if needed
    if models is None:
        models = list(metrics_dict.keys())
    else:
        # Ensure all requested models exist
        for model in models:
            if model not in metrics_dict:
                print(f"Warning: Model '{model}' not found in metrics data")
                models.remove(model)
    
    # Extract metrics for the specified feature for each model
    mape_values = []
    mae_values = []
    mse_values = []
    
    for model in models:
        if feature_name in metrics_dict[model]:
            mape_values.append(metrics_dict[model][feature_name]['MAPE'])
            mae_values.append(metrics_dict[model][feature_name]['MAE'])
            mse_values.append(metrics_dict[model][feature_name]['MSE'])
        else:
            print(f"Warning: Feature '{feature_name}' not found for model '{model}'")
            mape_values.append(0)
            mae_values.append(0)
            mse_values.append(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars with different patterns
    mape_bars = ax.bar(r1, mape_values, width=barWidth, label='MAPE (%)', 
                      color='#f0f0f0', edgecolor='black', linewidth=1)
    
    mae_bars = ax.bar(r2, mae_values, width=barWidth, label='MAE',
                     color='black', edgecolor='black', linewidth=1)
    
    mse_bars = ax.bar(r3, mse_values, width=barWidth, label='MSE',
                     color='white', edgecolor='black', linewidth=1, hatch='xxx')
    
    # Add grid lines
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars for MAPE
    for bar, value in zip(mape_bars, mape_values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.1,
                f"{value:.1f}%", ha='center', va='bottom', 
                fontsize=9, rotation=0)
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Add labels and title
    if config.VISUALIZATION_CONFIG['show_titles']:
        ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Error Value')
    
    # Add y-axis grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis tick labels
    plt.xticks([r + barWidth for r in range(len(models))], models)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if savepath:
        # Ensure directory exists
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()

def plot_all_industries_mape(industry_metrics, feature_name='y', savepath=None, title="Industry Model Performance Comparison (MAPE %)"):
    """
    Create a single bar chart showing MAPE percentages for all industries, sorted from best to worst.
    
    Args:
        industry_metrics (dict): Dictionary mapping industry names to metric dictionaries
                               Format: {industry_name: {'feature_name': {'MSE': val, 'MAE': val, 'MAPE': val}}}
        feature_name (str): The feature to plot metrics for (default: 'y')
        savepath (str): Path to save the figure
        title (str): Title for the plot
    """
    set_academic_style()
    
    # Extract MAPE values for each industry
    industries = []
    mape_values = []
    
    for industry, metrics in industry_metrics.items():
        if feature_name in metrics:
            industries.append(industry)
            mape_values.append(metrics[feature_name]['MAPE'])
    
    # Sort industries by MAPE (ascending - lower is better)
    sorted_indices = np.argsort(mape_values)
    industries = [industries[i] for i in sorted_indices]
    mape_values = [mape_values[i] for i in sorted_indices]
    
    # Create the plot - wider and shorter for double column in paper
    fig, ax = plt.subplots(figsize=(16, 6))  # Wider and shorter for double column
    
    # Create the bars with alternating colors
    bar_colors = ['#f0f0f0', '#d9d9d9']  # Light gray alternating with slightly darker gray
    
    bars = ax.bar(industries, mape_values, 
                 color=[bar_colors[i % 2] for i in range(len(industries))],
                 edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, mape_values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.1,
                f"{value:.1f}%", ha='center', va='bottom', 
                fontsize=9, rotation=90)
    
    # Add title and labels
    if config.VISUALIZATION_CONFIG['show_titles']:
        ax.set_title(title)
    ax.set_ylabel('MAPE (%)')
    
    # Add y-axis grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Set a reasonable y-axis limit based on the data
    max_value = max(mape_values)
    plt.ylim(0, max_value * 1.2)  # Add 20% padding at the top
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if savepath:
        # Ensure directory exists
        Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(savepath, dpi=config.VISUALIZATION_CONFIG['dpi'], bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()
