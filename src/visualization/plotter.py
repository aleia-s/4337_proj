import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, val_losses, savepath=None):
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_predictions(dates, y_true, y_pred, feature_names, n_plots=3, savepath=None):
    """
    Simple multi‐plot of actual vs. predicted for the first n_plots features.
    """
    import numpy as np
    idxs = np.arange(min(n_plots, y_true.shape[1]))
    fig, axes = plt.subplots(len(idxs), 1, figsize=(8, 3*len(idxs)))
    if len(idxs)==1:
        axes=[axes]
    for ax, i in zip(axes, idxs):
        ax.plot(dates, y_true[:,i],  label=f"Actual {feature_names[i]}")
        ax.plot(dates, y_pred[:,i],  "--", label=f"Pred {feature_names[i]}")
        ax.legend()
        ax.set_ylabel(feature_names[i])
    plt.xlabel("Time")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()
    plt.close()

def print_metrics(metrics: dict):
    """
    Pretty‐print the metrics dict:
      { feature: {'MSE':…, 'MAE':…, 'MAPE':…}, … }
    """
    print("\nFeature‐wise metrics:")
    for feat, vals in metrics.items():
        print(f" • {feat:25}  "
              f"MSE={vals['MSE']:.4f}, "
              f"MAE={vals['MAE']:.4f}, "
              f"MAPE={vals['MAPE']:.2f}%")
