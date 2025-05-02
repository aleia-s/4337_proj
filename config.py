import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()

# Default data file options
DATA_FILES = {
    "unemployment": "OLD_FILES/multivariate_unemployment_LSTNet.csv",
    "industries": "data/data.csv"
}

# Set which dataset to use - change this to switch between datasets
ACTIVE_DATASET = "industries"  # Options: "unemployment" or "industries"

# Load config from file if exists
try:
    _cfg = json.loads((PROJECT_ROOT / "data" / "config.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    _cfg = {
        "start_year": 2006,
        "end_year": 2024,
        "api_key": "",
        "series_codes_file": "data/bls_series_codes.json",
        "data_dir": "data",
        "models_dir": "results/models",
        "visualizations_dir": "visualizations",
        "output_csv": "data.csv"
    }

BLS_PARAMS = {
    "start_year": _cfg["start_year"],
    "end_year":   _cfg["end_year"],
    "api_key":    _cfg.get("api_key", ""),
    "series_codes_file": PROJECT_ROOT / _cfg.get("series_codes_file", "data/bls_series_codes.json"),
    "output_csv":        PROJECT_ROOT / _cfg["data_dir"] / _cfg["output_csv"],
}

DATA_CONFIG = {
    "data_dir":       PROJECT_ROOT / _cfg["data_dir"],
    "models_dir":     PROJECT_ROOT / _cfg["models_dir"],
    "visualizations_dir": PROJECT_ROOT / _cfg.get("visualizations_dir", "visualizations"),
    "default_data_file": DATA_FILES[ACTIVE_DATASET],
}


# LSTNet / model‐training parameters 
MODEL_CONFIG = {
    'cnn_kernel_size': 6,     # CNN kernel size
    'rnn_hidden_size': 100,   # RNN hidden size
    'skip_size': 24,          # Skip size for skip RNN
    'skip_hidden_size': 5,    # Skip RNN hidden size
    'highway_window': 24,     # Size of highway window
}

TRAINING_CONFIG = {
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.001,
    'sequence_length': 28,  # Number of time steps to look back
    'test_size': 0.2,       # Fraction of data to use for testing
    'val_size': 0.2,        # Fraction of remaining data to use for validation
}

VISUALIZATION_CONFIG = {
    'figure_size':        (10,5),
    'dpi':                 300,
    'font_family':        'Times New Roman',
    'font_size':           10,
    'title_size':          12,
    'label_size':          10,
    'tick_size':            8,
    'legend_size':          8,
    'colors': {
      'actual':   'black',
      'pred':     'red',
      'mape_bar': '#f0f0f0',
      'mae_bar':   'black',
      'mse_bar':   'white',
    },
    'line_styles': {
      'actual':   '-',
      'pred':     '--',
    },
    'line_width':       1.5,
    'grid_alpha':       0.7,
    'grid_style':      '--',
    'bar_width':       0.25,
    'bar_patterns': {
      'mape': '',
      'mae':  '',
      'mse':  'xxx',
    },
    'show_titles':        True,
    'show_value_labels':  True,
    'value_label_decimals': {
      'mape': 1,
      'mae':  3,
      'mse':  3
    },
    'label_offset':      0.1,
    'margin_top':        0.2,
    'x_rotation':        45,
    'legend_loc':      'upper right',
}
